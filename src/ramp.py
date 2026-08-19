#!/usr/bin/env python3

import math
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Int32


class RampNode(Node):
    """
    Stage 8 rampa + lazer görevi.

    Çalışma sırası:
    1. /teknofest/stage_id == 8 olana kadar bekler (APPROACH_RAMP).
    2. Rampaya çıkış algılandığında (pitch >= 8°) 2s durur (RAMP_UP_STOP).
    3. Rampayı tırmanır (CLIMBING_RAMP).
    4. Düzlüğe ulaşıldığında veya STOP tabelası görüldüğünde durur (STOPPING_AT_TOP - 2s).
    5. 1.2 saniye lazer yakar (LASER_ON).
    6. Rampadan aşağı iner (DESCENDING_RAMP).
    7. İniş tamamlandığında 2s durur (RAMP_DOWN_STOP).
    8. /teknofest/release = 8 yayınlar (DONE).
    """

    def __init__(self):
        super().__init__("ramp")

        self.declare_parameter("active_stage", 8)
        self.declare_parameter("initial_stage", 0)
        self.declare_parameter("drive_speed", 0.35)

        self.declare_parameter("ramp_pitch_threshold_deg", 8.0)
        self.declare_parameter("flat_pitch_threshold_deg", 3.0)

        self.declare_parameter("stop_duration", 2.0)
        self.declare_parameter("laser_duration", 1.2)
        self.declare_parameter("flat_drive_duration", 1.0)
        self.declare_parameter("top_flat_drive_duration", 1.5)
        self.declare_parameter("ramp_up_drive_duration", 1.2)
        self.declare_parameter("descent_drive_duration", 2.0)
        self.declare_parameter("top_brake_duration", 0.35)
        self.declare_parameter("down_brake_duration", 2.0)

        self.active_stages = [8, 9, 10]
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
        self.top_flat_drive_duration = float(self.get_parameter("top_flat_drive_duration").value)
        self.ramp_up_drive_duration = float(self.get_parameter("ramp_up_drive_duration").value)
        self.descent_drive_duration = float(self.get_parameter("descent_drive_duration").value)
        self.top_brake_duration = float(self.get_parameter("top_brake_duration").value)
        self.down_brake_duration = float(self.get_parameter("down_brake_duration").value)

        self.current_stage = self.initial_stage
        self.pitch = 0.0
        self.odom_pos = (0.0, 0.0, 0.5)

        self.ramp_up_confirm_frames = 0
        self.flat_confirm_frames = 0
        self.descent_pitch_seen = False
        self.descent_flat_frames = 0

        self.state = (
            "APPROACH_RAMP"
            if self.current_stage in self.active_stages
            else "WAIT_STAGE"
        )
        self.state_start_time = self.now_seconds()

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/ramp/cmd_vel",
            10,
        )
        self.release_pub = self.create_publisher(
            Int32,
            "/teknofest/release",
            10,
        )
        self.parking_brake_pub = self.create_publisher(
            Bool,
            "/rover/parking_brake",
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

        self.timer = self.create_timer(
            0.05,
            self.control_loop,
        )

        self.get_logger().info(
            f"=== Stage 8-9-10 Ramp Node başladı (Hız={self.drive_speed} m/s, Durma={self.stop_duration}s) ==="
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

    def publish_release(self, stage_num):
        msg = Int32()
        msg.data = int(stage_num)
        self.release_pub.publish(msg)
        self.get_logger().info(f"Stage {stage_num} serbest bırakıldı (/teknofest/release={stage_num}).")

    def publish_parking_brake(self, enable: bool):
        msg = Bool()
        msg.data = bool(enable)
        self.parking_brake_pub.publish(msg)

    def reset_task(self):
        self.ramp_up_confirm_frames = 0
        self.flat_confirm_frames = 0
        self.descent_pitch_seen = False
        self.descent_flat_frames = 0

        self.publish_parking_brake(False)
        self.publish_velocity(0.0, 0.0)
        self.set_state("WAIT_STAGE")

    def stage_callback(self, msg):
        incoming_stage = int(msg.data)
        previous_stage = self.current_stage
        self.current_stage = incoming_stage

        if (
            self.current_stage in self.active_stages
            and previous_stage not in self.active_stages
        ):
            self.reset_task()
            self.set_state("APPROACH_RAMP")
            self.get_logger().info(f"Stage {self.current_stage} aktif: Rampa dizisi başlatıldı.")

        elif (
            self.current_stage not in self.active_stages
            and previous_stage in self.active_stages
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
        if self.state in ("START_DESCENT", "DESCENT_DRIVING"):
            if abs_pitch >= self.ramp_pitch_threshold or abs_pitch >= math.radians(5.0):
                self.descent_pitch_seen = True

    def control_loop(self):
        if self.current_stage not in self.active_stages:
            self.publish_velocity(0.0, 0.0)
            return

        if self.state == "APPROACH_RAMP":
            # Driving towards ramp
            self.publish_parking_brake(False)
            self.publish_velocity(linear_x=self.drive_speed)

            # Ramp climb detected (pitch >= threshold)
            if self.ramp_up_confirm_frames >= 2 or abs(self.pitch) >= self.ramp_pitch_threshold:
                self.set_state("DRIVING_UP_RAMP_MID")
                self.get_logger().warning(
                    f"Rampa çıkış eğimi algılandı (Pitch={math.degrees(self.pitch):.1f}°). "
                    f"Rampa ortasına tırmanılıyor ({self.ramp_up_drive_duration}s sürüş)..."
                )

        elif self.state == "DRIVING_UP_RAMP_MID":
            # Driving up the slope towards the middle of the ramp incline
            self.publish_parking_brake(False)
            self.publish_velocity(linear_x=self.drive_speed)

            if self.elapsed_in_state() >= self.ramp_up_drive_duration:
                self.publish_velocity(0.0, 0.0)
                self.set_state("RAMP_UP_STOP")
                self.get_logger().warning(
                    f"Rampa ortasına ulaşıldı. {self.stop_duration}s duruluyor."
                )

        elif self.state == "RAMP_UP_STOP":
            # Stop at ramp middle with active parking brake
            self.publish_parking_brake(True)
            self.publish_velocity(0.0, 0.0)

            if self.elapsed_in_state() >= self.stop_duration:
                self.set_state("CLIMBING_RAMP")
                self.get_logger().info("Rampa orta duruşu tamamlandı. Zirveye tırmanış devam ediyor.")

        elif self.state == "CLIMBING_RAMP":
            # Climbing ramp towards the top flat area
            self.publish_parking_brake(False)
            self.publish_velocity(linear_x=self.drive_speed)

            # Reached top flat area via IMU flat detection
            if self.flat_confirm_frames >= 4:
                self.set_state("DRIVING_ON_TOP_FLAT")
                self.get_logger().warning(
                    f"Rampa tepe düzlüğü algılandı. Durmadan önce {self.top_flat_drive_duration}s sürüşe devam ediliyor."
                )

        elif self.state == "DRIVING_ON_TOP_FLAT":
            # Continue driving forward on top flat platform for top_flat_drive_duration (1.5s) before stopping
            self.publish_parking_brake(False)
            self.publish_velocity(linear_x=self.drive_speed)

            if self.elapsed_in_state() >= self.top_flat_drive_duration:
                self.publish_velocity(0.0, 0.0)
                self.set_state("STOPPING_AT_TOP")
                self.publish_release(8)  # Stage 8 (Tırmanma) bitti -> Stage 9 (Tepe/Lazer)
                self.get_logger().warning("Tepe düzlüğü sürüşü tamamlandı (Stage 8 -> 9). Park freni aktif, target_detect bekleniyor.")

        elif self.state == "STOPPING_AT_TOP":
            # Engage parking brake to hold position while waiting for target_detect to complete Stage 9
            self.publish_parking_brake(True)
            self.publish_velocity(0.0, 0.0)

            # Target_detect handles laser firing and publishes release(9), moving stage to 10
            if self.current_stage == 10:
                self.descent_pitch_seen = False
                self.set_state("START_DESCENT")
                self.get_logger().info("Target detect lazer görevini tamamladı (Stage 9 -> 10). İniş sürüşü başlatılıyor.")

        elif self.state == "START_DESCENT":
            # Driving forward off top platform towards descent slope
            self.publish_parking_brake(False)
            self.publish_velocity(linear_x=self.drive_speed)

            # Descent slope entered (pitch >= 5 deg or threshold)
            if self.descent_pitch_seen:
                self.set_state("DESCENT_DRIVING")
                self.get_logger().warning(
                    f"[İNİŞ EĞİMİ ALGILANDI] Rampa iniş eğimine girildi (Pitch={math.degrees(self.pitch):.1f}°). "
                    f"{self.descent_drive_duration}s boyunca rampadan iniliyor..."
                )

        elif self.state == "DESCENT_DRIVING":
            # Driving forward down the ramp for descent_drive_duration
            self.publish_parking_brake(False)
            self.publish_velocity(linear_x=self.drive_speed)

            if self.elapsed_in_state() >= self.descent_drive_duration:
                self.set_state("RAMP_DOWN_STOP")
                self.get_logger().warning(
                    f"Rampa iniş sürüşü ({self.descent_drive_duration}s) tamamlandı. Park freni aktif ediliyor ({self.down_brake_duration}s)."
                )

        elif self.state == "RAMP_DOWN_STOP":
            # Engage parking brake to cancel downhill momentum and hold position
            elapsed = self.elapsed_in_state()
            self.publish_parking_brake(True)
            self.publish_velocity(0.0, 0.0)

            if elapsed >= max(self.stop_duration, self.down_brake_duration):
                self.set_state("DONE")
                self.publish_release(10)  # Stage 10 (İniş/Son) bitti
                self.get_logger().info("Rampa iniş park frenlemesi ve duruşu tamamlandı. Stage 10 görevi bitti.")

        elif self.state == "DONE":
            self.publish_parking_brake(False)
            self.publish_velocity(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = RampNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_velocity(linear_x=0.0)
            node.publish_parking_brake(False)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
