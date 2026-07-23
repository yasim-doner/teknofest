#!/usr/bin/env python3

import math
from enum import Enum, auto

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Int32, String


class LaserTaskState(Enum):
    WAIT_STAGE = auto()
    APPROACH_RAMP = auto()
    ASCENDING_RAMP = auto()
    SEARCH_STOP = auto()
    STOPPING = auto()
    LASER_ON = auto()
    DONE = auto()


class LaserTarget(Node):
    """
    Stage 8 rampa + lazer görevi.

    Çalışma sırası:
    1. /teknofest/stage_id == 8 olana kadar bekler.
    2. IMU pitch açısıyla rampaya çıkıldığını anlar.
    3. Rampadan sonra zemin yeniden düzleşince yavaşlar.
    4. STOP tabelası görülürse hemen; görülmezse kısa güvenlik süresi
       sonunda roverı durdurur.
    5. Roverı en az 2 saniye tamamen durdurur.
    6. Lazeri en az 1 saniye açık tutar.
    7. /teknofest/laser_complete = True yayınlar.

    Not:
    - Bu node lazerin Gazebo'daki görselini kendisi oluşturmaz.
    - /teknofest/laser_on Bool topic'ini yayımlar.
      Simülatördeki lazer modeli veya lazer controller bu topic'e abone olmalıdır.
    - Hareket komutunu /laser_target/cmd_vel üzerinden yayımlar.
      cmd_switch.py stage 8'de bu topic'i /rover/cmd_vel'e aktarmalıdır.
    """

    def __init__(self):
        super().__init__("laser_target")

        self.declare_parameter("active_stage", 8)
        self.declare_parameter("initial_stage", 0)

        self.declare_parameter("approach_speed", 0.35)
        self.declare_parameter("ramp_speed", 0.45)
        self.declare_parameter("ramp_boost_speed", 0.55)
        self.declare_parameter("ramp_boost_delay", 2.0)
        self.declare_parameter("search_speed", 0.08)

        self.declare_parameter("ramp_pitch_threshold_deg", 8.0)
        self.declare_parameter("flat_pitch_threshold_deg", 3.0)

        self.declare_parameter("ramp_confirm_frames", 5)
        self.declare_parameter("flat_confirm_frames", 12)

        self.declare_parameter("stop_sign_wait_timeout", 2.0)
        self.declare_parameter("stop_duration", 2.2)
        self.declare_parameter("laser_duration", 1.2)

        self.active_stage = int(self.get_parameter("active_stage").value)
        self.initial_stage = int(self.get_parameter("initial_stage").value)

        self.approach_speed = float(self.get_parameter("approach_speed").value)
        self.ramp_speed = float(self.get_parameter("ramp_speed").value)
        self.ramp_boost_speed = float(
            self.get_parameter("ramp_boost_speed").value
        )
        self.ramp_boost_delay = max(
            0.0,
            float(self.get_parameter("ramp_boost_delay").value),
        )
        self.search_speed = float(self.get_parameter("search_speed").value)

        self.ramp_pitch_threshold = math.radians(
            float(self.get_parameter("ramp_pitch_threshold_deg").value)
        )
        self.flat_pitch_threshold = math.radians(
            float(self.get_parameter("flat_pitch_threshold_deg").value)
        )

        self.ramp_confirm_frames = max(
            1,
            int(self.get_parameter("ramp_confirm_frames").value),
        )
        self.flat_confirm_frames = max(
            1,
            int(self.get_parameter("flat_confirm_frames").value),
        )

        self.stop_sign_wait_timeout = float(
            self.get_parameter("stop_sign_wait_timeout").value
        )
        self.stop_duration = max(
            2.0,
            float(self.get_parameter("stop_duration").value),
        )
        self.laser_duration = max(
            1.0,
            float(self.get_parameter("laser_duration").value),
        )

        self.current_stage = self.initial_stage
        self.state = (
            LaserTaskState.APPROACH_RAMP
            if self.current_stage == self.active_stage
            else LaserTaskState.WAIT_STAGE
        )

        self.pitch = 0.0
        self.stop_detected = False

        self.ramp_frame_count = 0
        self.flat_frame_count = 0

        self.state_start_time = self.now_seconds()
        self.task_complete_sent = False
        self.last_ramp_log_time = -1000.0

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
        self.complete_pub = self.create_publisher(
            Bool,
            "/teknofest/laser_complete",
            10,
        )
        self.release_pub = self.create_publisher(
            Int32,
            "/teknofest/release",
            10,
        )
        self.state_pub = self.create_publisher(
            String,
            "/teknofest/laser_state",
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
            "=== Stage 8 Laser Target Node başladı ==="
        )
        self.get_logger().info(
            f"Aktif stage={self.active_stage}, "
            f"başlangıç stage={self.current_stage}, "
            f"durma={self.stop_duration:.1f}s, "
            f"lazer={self.laser_duration:.1f}s"
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

        self.get_logger().info(
            f"Durum değişti: {old_state.name} -> {new_state.name}"
        )

    def publish_velocity(self, linear_x=0.0, angular_z=0.0):
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(cmd)

    def publish_laser(self, is_on):
        msg = Bool()
        msg.data = bool(is_on)
        self.laser_pub.publish(msg)

    def publish_complete(self, is_complete):
        msg = Bool()
        msg.data = bool(is_complete)
        self.complete_pub.publish(msg)

    def publish_state(self):
        msg = String()
        msg.data = self.state.name
        self.state_pub.publish(msg)

    def reset_task(self):
        self.ramp_frame_count = 0
        self.flat_frame_count = 0
        self.stop_detected = False
        self.task_complete_sent = False

        self.publish_velocity()
        self.publish_laser(False)
        self.publish_complete(False)

        self.set_state(LaserTaskState.WAIT_STAGE)

    def stage_callback(self, msg):
        incoming_stage = int(msg.data)

        # stage:=8 ile doğrudan test sırasında sign detector'ın geçici
        # 0 veya daha küçük bir stage üretmesi görevi kapatmasın.
        if (
            self.initial_stage == self.active_stage
            and self.state not in (
                LaserTaskState.WAIT_STAGE,
                LaserTaskState.DONE,
            )
            and incoming_stage < self.active_stage
        ):
            return

        previous_stage = self.current_stage
        self.current_stage = incoming_stage

        if (
            self.current_stage == self.active_stage
            and previous_stage != self.active_stage
        ):
            self.ramp_frame_count = 0
            self.flat_frame_count = 0
            self.stop_detected = False
            self.task_complete_sent = False

            self.publish_laser(False)
            self.publish_complete(False)

            self.set_state(LaserTaskState.APPROACH_RAMP)

            self.get_logger().info(
                "Stage 8 aktif: rampa aranıyor."
            )

        elif (
            self.current_stage != self.active_stage
            and previous_stage == self.active_stage
        ):
            self.get_logger().info(
                "Stage 8 bitti veya değişti; laser_target beklemeye geçti."
            )
            self.reset_task()

    def imu_callback(self, msg):
        self.pitch = self.quaternion_to_pitch(msg)
        abs_pitch = abs(self.pitch)

        if abs_pitch >= self.ramp_pitch_threshold:
            self.ramp_frame_count += 1
        else:
            self.ramp_frame_count = 0

        if (
            self.state == LaserTaskState.ASCENDING_RAMP
            and abs_pitch <= self.flat_pitch_threshold
        ):
            self.flat_frame_count += 1
        else:
            self.flat_frame_count = 0

    def stop_callback(self, msg):
        self.stop_detected = bool(msg.data)

    def control_loop(self):
        self.publish_state()

        if self.current_stage != self.active_stage:
            self.publish_velocity()
            self.publish_laser(False)
            return

        if self.state == LaserTaskState.APPROACH_RAMP:
            self.publish_laser(False)
            self.publish_velocity(linear_x=self.approach_speed)

            if self.ramp_frame_count >= self.ramp_confirm_frames:
                self.set_state(LaserTaskState.ASCENDING_RAMP)
                self.get_logger().info(
                    f"Rampa algılandı. Pitch={math.degrees(self.pitch):.1f} derece"
                )

        elif self.state == LaserTaskState.ASCENDING_RAMP:
            self.publish_laser(False)

            # Rampanın ilk bölümünde kontrollü hız kullan.
            # İki saniye sonra hâlâ eğimdeyse tekerleklerin geri kaymasını
            # engellemek için kısa bir hız desteği uygula.
            ramp_elapsed = self.elapsed_in_state()

            if ramp_elapsed >= self.ramp_boost_delay:
                commanded_speed = self.ramp_boost_speed
            else:
                commanded_speed = self.ramp_speed

            self.publish_velocity(
                linear_x=commanded_speed
            )

            # Terminali doldurmadan rampadaki durumu saniyede bir göster.
            now = self.now_seconds()
            if now - self.last_ramp_log_time >= 1.0:
                self.last_ramp_log_time = now
                self.get_logger().info(
                    f"Rampa çıkılıyor: "
                    f"pitch={math.degrees(self.pitch):.1f} derece, "
                    f"hız={commanded_speed:.2f} m/s"
                )

            if self.flat_frame_count >= self.flat_confirm_frames:
                self.set_state(LaserTaskState.SEARCH_STOP)
                self.get_logger().info(
                    "Rampa bitti, düz zemin algılandı. STOP tabelası bekleniyor."
                )

        elif self.state == LaserTaskState.SEARCH_STOP:
            self.publish_laser(False)

            if self.stop_detected:
                self.publish_velocity()
                self.set_state(LaserTaskState.STOPPING)
                self.get_logger().warning(
                    "STOP tabelası algılandı. Rover tamamen durduruldu."
                )

            elif self.elapsed_in_state() >= self.stop_sign_wait_timeout:
                self.publish_velocity()
                self.set_state(LaserTaskState.STOPPING)
                self.get_logger().warning(
                    "STOP tabelası zamanında algılanmadı; güvenlik duruşu başlatıldı."
                )

            else:
                self.publish_velocity(linear_x=self.search_speed)

        elif self.state == LaserTaskState.STOPPING:
            self.publish_velocity()
            self.publish_laser(False)

            if self.elapsed_in_state() >= self.stop_duration:
                self.set_state(LaserTaskState.LASER_ON)
                self.get_logger().warning(
                    "Duruş tamamlandı. Lazer açıldı."
                )

        elif self.state == LaserTaskState.LASER_ON:
            self.publish_velocity()
            self.publish_laser(True)

            if self.elapsed_in_state() >= self.laser_duration:
                self.publish_laser(False)
                self.set_state(LaserTaskState.DONE)
                self.get_logger().info(
                    "Lazer görevi tamamlandı."
                )

        elif self.state == LaserTaskState.DONE:
            self.publish_velocity()
            self.publish_laser(False)

            if not self.task_complete_sent:
                self.publish_complete(True)

                release_msg = Int32()
                release_msg.data = self.active_stage
                self.release_pub.publish(release_msg)

                self.task_complete_sent = True
                self.get_logger().info(
                    "Stage 8 kontrolü cmd_switch'e bırakıldı."
                )

        else:
            self.publish_velocity()
            self.publish_laser(False)


def main(args=None):
    rclpy.init(args=args)
    node = LaserTarget()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            node.publish_velocity()
            node.publish_laser(False)

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()