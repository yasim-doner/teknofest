#!/usr/bin/env python3

import math
import rclpy
from geometry_msgs.msg import PointStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Int32


def calculate_turret_angles(p_laser, p_target):
    """
    Calculates precise Yaw and Pitch angles in degrees.
    p_laser:  (x, y, z) tuple/list
    p_target: (x, y, z) tuple/list
    """
    dx = p_target[0] - p_laser[0]
    dy = p_target[1] - p_laser[1]
    dz = p_target[2] - p_laser[2]

    # Horizontal distance in XY plane
    d_xy = math.hypot(dx, dy)

    # Calculate angles in radians
    yaw_rad = math.atan2(dy, dx)
    pitch_rad = math.atan2(dz, d_xy)

    # Convert to degrees
    yaw_deg = math.degrees(yaw_rad)
    pitch_deg = math.degrees(pitch_rad)

    return yaw_deg, pitch_deg


class LaserTarget(Node):
    """
    Stage 8 rampa + lazer görevi (Sadeleştirilmiş).

    Çalışma sırası:
    1. /teknofest/stage_id == 8 olana kadar bekler.
    2. Sabit bir hızla (drive_speed) doğrudan düz ilerler.
    3. IMU ile rampaya çıktığını ve ardından düzlüğe ulaştığını tespit eder.
    4. Düzlüğe ulaştığında veya STOP tabelası görüldüğünde durur.
    5. 2 saniye durup 1.2 saniye lazer yakar.
    6. Lazer açıkken kameradan gelen hedef noktasına (/teknofest/target_point) göre
       hesaplanan yaw ve pitch (derece) açılarını /laser_angle üzerinden yayınlar.
    7. Lazer bittiğinde /teknofest/laser_on = False yapar ve /teknofest/release = 8 yayınlar.
    """

    def __init__(self):
        super().__init__("laser_target")

        self.declare_parameter("active_stage", 8)
        self.declare_parameter("initial_stage", 0)
        self.declare_parameter("drive_speed", 0.35)

        self.declare_parameter("ramp_pitch_threshold_deg", 8.0)
        self.declare_parameter("flat_pitch_threshold_deg", 3.0)

        self.declare_parameter("stop_duration", 2.0)
        self.declare_parameter("laser_duration", 1.2)
        self.declare_parameter("flat_drive_duration", 1.0)

        self.active_stage = int(self.get_parameter("active_stage").value)
        self.initial_stage = int(self.get_parameter("initial_stage").value)
        self.drive_speed = float(self.get_parameter("drive_speed").value)

        self.ramp_pitch_threshold = math.radians(
            float(self.get_parameter("ramp_pitch_threshold_deg").value)
        )
        self.flat_pitch_threshold = math.radians(
            float(self.get_parameter("flat_pitch_threshold_deg").value)
        )

        self.stop_duration = float(self.get_parameter("stop_duration").value)
        self.laser_duration = float(self.get_parameter("laser_duration").value)
        self.flat_drive_duration = float(self.get_parameter("flat_drive_duration").value)

        self.target_pos = None  # Kameradan dinamik tespit edilir (/teknofest/target_point)

        self.current_stage = self.initial_stage
        self.pitch = 0.0
        self.odom_pos = (0.0, 0.0, 0.5)

        self.stop_detected = False
        self.ramp_detected = False
        self.flat_confirm_frames = 0
        self.flat_drive_start_time = None
        self.state = (
            "DRIVING"
            if self.current_stage == self.active_stage
            else "WAIT_STAGE"
        )
        self.state_start_time = self.now_seconds()

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/laser_target/cmd_vel",
            10,
        )
        self.laser_pub = self.create_publisher(
            Bool,
            "/teknofest/laser_on",
            10,
        )
        self.laser_angle_pub = self.create_publisher(
            Vector3,
            "/laser_angle",
            10,
        )
        self.release_pub = self.create_publisher(
            Int32,
            "/teknofest/release",
            10,
        )

        self.stage_sub = self.create_subscription(
            Int32,
            "/teknofest/stage_id",
            self.stage_callback,
            10,
        )
        self.imu_sub = self.create_subscription(
            Imu,
            "/rover/imu",
            self.imu_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            "/rover/odom",
            self.odom_callback,
            10,
        )
        self.target_sub = self.create_subscription(
            PointStamped,
            "/teknofest/target_point",
            self.target_point_callback,
            10,
        )
        self.stop_sub = self.create_subscription(
            Bool,
            "/teknofest/stop_detected",
            self.stop_callback,
            10,
        )

        self.timer = self.create_timer(
            0.05,
            self.control_loop,
        )

        self.get_logger().info(
            f"=== Stage 8 Laser Target Node başladı (Hız={self.drive_speed} m/s) ==="
        )

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1_000_000_000.0

    @staticmethod
    def quaternion_to_pitch(msg):
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        sin_pitch = 2.0 * (w * y - z * x)
        sin_pitch = max(-1.0, min(1.0, sin_pitch))
        return math.asin(sin_pitch)

    def elapsed_in_state(self):
        return self.now_seconds() - self.state_start_time

    def set_state(self, new_state):
        if new_state == self.state:
            return

        old_state = self.state
        self.state = new_state
        self.state_start_time = self.now_seconds()
        self.get_logger().info(f"Durum değişti: {old_state} -> {new_state}")

    def publish_velocity(self, linear_x=0.0, angular_z=0.0):
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(cmd)

    def publish_laser(self, is_on):
        msg = Bool()
        msg.data = bool(is_on)
        self.laser_pub.publish(msg)

    def publish_laser_angle(self):
        if self.target_pos is None:
            self.get_logger().warning(
                "Kameradan henüz hedef noktası (/teknofest/target_point) alınmadı!",
                throttle_duration_sec=2.0,
            )
            return

        yaw_deg, pitch_deg = calculate_turret_angles(self.odom_pos, self.target_pos)
        angle_msg = Vector3()
        angle_msg.x = float(yaw_deg)
        angle_msg.y = float(pitch_deg)
        angle_msg.z = 0.0
        self.laser_angle_pub.publish(angle_msg)

    def target_point_callback(self, msg):
        self.target_pos = (
            float(msg.point.x),
            float(msg.point.y),
            float(msg.point.z),
        )

    def reset_task(self):
        self.stop_detected = False
        self.ramp_detected = False
        self.flat_confirm_frames = 0
        self.flat_drive_start_time = None

        self.publish_velocity()
        self.publish_laser(False)
        self.set_state("WAIT_STAGE")

    def stage_callback(self, msg):
        incoming_stage = int(msg.data)
        previous_stage = self.current_stage
        self.current_stage = incoming_stage

        if (
            self.current_stage == self.active_stage
            and previous_stage != self.active_stage
        ):
            self.reset_task()
            self.set_state("DRIVING")
            self.get_logger().info("Stage 8 aktif: Sürüş başladı.")

        elif (
            self.current_stage != self.active_stage
            and previous_stage == self.active_stage
        ):
            self.reset_task()

    def odom_callback(self, msg):
        self.odom_pos = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            float(msg.pose.pose.position.z) + 0.5,
        )

    def imu_callback(self, msg):
        self.pitch = self.quaternion_to_pitch(msg)
        abs_pitch = abs(self.pitch)

        if abs_pitch >= self.ramp_pitch_threshold:
            self.ramp_detected = True

        if self.ramp_detected and abs_pitch <= self.flat_pitch_threshold:
            self.flat_confirm_frames += 1
        else:
            self.flat_confirm_frames = 0

    def stop_callback(self, msg):
        self.stop_detected = bool(msg.data)

    def control_loop(self):
        if self.current_stage != self.active_stage:
            self.publish_velocity()
            self.publish_laser(False)
            return

        if self.state == "DRIVING":
            self.publish_laser(False)
            self.publish_velocity(linear_x=self.drive_speed)

            if self.stop_detected or self.flat_confirm_frames >= 5:
                if self.flat_drive_start_time is None:
                    self.flat_drive_start_time = self.now_seconds()
                    reason = "STOP tabelası" if self.stop_detected else "Rampa bitti, düz zemin"
                    self.get_logger().warning(
                        f"{reason} algılandı. {self.flat_drive_duration:.1f}s daha sürüşe devam ediliyor."
                    )

                if self.now_seconds() - self.flat_drive_start_time >= self.flat_drive_duration:
                    self.publish_velocity(linear_x=0.0)
                    self.set_state("STOPPING")
                    self.get_logger().warning(
                        f"Düzlükteki {self.flat_drive_duration:.1f}s sürüş tamamlandı. Rover durduruldu."
                    )

        elif self.state == "STOPPING":
            self.publish_velocity(linear_x=0.0)
            self.publish_laser(False)

            if self.elapsed_in_state() >= self.stop_duration:
                self.set_state("LASER_ON")
                self.get_logger().warning("Duruş süresi doldu. Lazer açılıyor.")

        elif self.state == "LASER_ON":
            self.publish_velocity(linear_x=0.0)
            self.publish_laser(True)
            self.publish_laser_angle()

            if self.elapsed_in_state() >= self.laser_duration:
                self.publish_laser(False)
                self.set_state("DONE")
                self.get_logger().info("Lazer süresi doldu. Görev tamamlandı.")

        elif self.state == "DONE":
            self.publish_velocity(linear_x=0.0)
            self.publish_laser(False)

            release_msg = Int32()
            release_msg.data = self.active_stage
            self.release_pub.publish(release_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaserTarget()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_velocity(linear_x=0.0)
            node.publish_laser(False)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
