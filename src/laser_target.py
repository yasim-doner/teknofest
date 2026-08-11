#!/usr/bin/env python3

import math
import rclpy
from geometry_msgs.msg import PointStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
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
    Stage 8 rampa + lazer görevi.

    Çalışma sırası:
    1. /teknofest/stage_id == 8 olana kadar bekler (APPROACH_RAMP).
    2. Rampaya çıkış algılandığında (pitch >= 8°) 2s durur (RAMP_UP_STOP).
    3. Rampayı tırmanır (CLIMBING_RAMP).
    4. Düzlüğe ulaşıldığında veya STOP tabelası görüldüğünde durur (STOPPING_AT_TOP - 2s).
    5. 1.2 saniye lazer yakar (LASER_ON) ve /laser_angle yayınlar.
    6. Rampadan aşağı iner (DESCENDING_RAMP).
    7. İniş tamamlandığında 2s durur (RAMP_DOWN_STOP).
    8. /teknofest/release = 8 yayınlar (DONE).
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

        self.ramp_up_confirm_frames = 0
        self.flat_confirm_frames = 0
        self.descent_pitch_seen = False
        self.descent_flat_frames = 0

        self.state = (
            "APPROACH_RAMP"
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
            qos_profile_sensor_data,
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

        self.timer = self.create_timer(
            0.05,
            self.control_loop,
        )

        self.get_logger().info(
            f"=== Stage 8 Laser Target Node başladı (Hız={self.drive_speed} m/s, Durma={self.stop_duration}s) ==="
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
        self.ramp_up_confirm_frames = 0
        self.flat_confirm_frames = 0
        self.descent_pitch_seen = False
        self.descent_flat_frames = 0

        self.publish_velocity(0.0, 0.0)
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
            self.set_state("APPROACH_RAMP")
            self.get_logger().info("Stage 8 aktif: Rampa yaklaşımı başladı.")

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

        # Ramp up pitch check
        if abs_pitch >= self.ramp_pitch_threshold:
            self.ramp_up_confirm_frames += 1
        else:
            self.ramp_up_confirm_frames = 0

        # Flat pitch check after climbing
        if self.state == "CLIMBING_RAMP" and abs_pitch <= self.flat_pitch_threshold:
            self.flat_confirm_frames += 1
        else:
            self.flat_confirm_frames = 0

        # Flat pitch check after descending
        if self.state == "DESCENDING_RAMP":
            if abs_pitch >= self.ramp_pitch_threshold:
                self.descent_pitch_seen = True

            if self.descent_pitch_seen and abs_pitch <= self.flat_pitch_threshold:
                self.descent_flat_frames += 1
            else:
                self.descent_flat_frames = 0

    def control_loop(self):
        if self.current_stage != self.active_stage:
            self.publish_velocity(0.0, 0.0)
            self.publish_laser(False)
            return

        if self.state == "APPROACH_RAMP":
            # Driving towards ramp
            self.publish_laser(False)
            self.publish_velocity(linear_x=self.drive_speed)

            # Ramp climb detected (pitch >= threshold)
            if self.ramp_up_confirm_frames >= 2 or abs(self.pitch) >= self.ramp_pitch_threshold:
                self.publish_velocity(0.0, 0.0)
                self.set_state("RAMP_UP_STOP")
                self.get_logger().warning(
                    f"Rampa çıkışı algılandı (Pitch={math.degrees(self.pitch):.1f}°). {self.stop_duration}s duruluyor."
                )

        elif self.state == "RAMP_UP_STOP":
            # 2 second stop at ramp start (climbing)
            self.publish_velocity(0.0, 0.0)
            self.publish_laser(False)

            if self.elapsed_in_state() >= self.stop_duration:
                self.set_state("CLIMBING_RAMP")
                self.get_logger().info("Rampa çıkış duruşu tamamlandı. Tırmanış başladı.")

        elif self.state == "CLIMBING_RAMP":
            # Climbing ramp towards the top flat area
            self.publish_laser(False)
            self.publish_velocity(linear_x=self.drive_speed)

            # Reached top flat area via IMU flat detection
            if self.flat_confirm_frames >= 4:
                self.publish_velocity(0.0, 0.0)
                self.set_state("STOPPING_AT_TOP")
                self.get_logger().warning("Rampa tepe düzlüğü algılandı. Atış için duruluyor.")

        elif self.state == "STOPPING_AT_TOP":
            # 2 second stop at top for laser task
            self.publish_velocity(0.0, 0.0)
            self.publish_laser(False)

            if self.elapsed_in_state() >= self.stop_duration:
                self.set_state("LASER_ON")
                self.get_logger().warning("Atış duruş süresi doldu. Lazer açılıyor.")

        elif self.state == "LASER_ON":
            # Laser on for laser_duration (1.2s) and publishing target angle
            self.publish_velocity(0.0, 0.0)
            self.publish_laser(True)
            self.publish_laser_angle()

            if self.elapsed_in_state() >= self.laser_duration:
                self.publish_laser(False)
                self.descent_pitch_seen = False
                self.descent_flat_frames = 0
                self.set_state("DESCENDING_RAMP")
                self.get_logger().info("Lazer görevi tamamlandı. Rampadan iniliyor.")

        elif self.state == "DESCENDING_RAMP":
            # Driving forward down the ramp
            self.publish_laser(False)
            self.publish_velocity(linear_x=self.drive_speed)

            # Check if descent slope finished or timeout reached
            if self.descent_flat_frames >= 4 or self.elapsed_in_state() >= 4.0:
                self.publish_velocity(0.0, 0.0)
                self.set_state("RAMP_DOWN_STOP")
                self.get_logger().warning(f"Rampadan iniş tamamlandı. {self.stop_duration}s duruluyor.")

        elif self.state == "RAMP_DOWN_STOP":
            # 2 second stop after descending ramp
            self.publish_velocity(0.0, 0.0)
            self.publish_laser(False)

            if self.elapsed_in_state() >= self.stop_duration:
                self.set_state("DONE")
                self.get_logger().info("Rampa iniş duruşu tamamlandı. Stage 8 görevi bitti.")

        elif self.state == "DONE":
            self.publish_velocity(0.0, 0.0)
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
