#!/usr/bin/env python3

import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Int32, String


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class CmdSwitchNode(Node):
    """
    Aktif stage'e göre doğru cmd_vel kaynağını /rover/cmd_vel'e aktarır.
    Ayrıca parkurdaki stage ilerlemesini, geofence ve odometri fallback
    mantığını merkezi olarak yönetir.

    Stage 5  -> /cone_avoid/cmd_vel
    Stage 6  -> /dynamic_obstacle/cmd_vel
    Stage 8  -> /ramp/cmd_vel
    Diğerleri -> /fallow_corridor/cmd_vel
    """

    STAGE_POSITIONS = {
        1: (16.0, 0.0),
        2: (11.0, 0.0),
        3: (6.0, 0.0),
        4: (2.0, 10.0),
        5: (7.0, 10.0),
        6: (14.0, 10.0),
        7: (18.0, 20.0),
        8: (11.5, 20.0),
        9: (7.8, 20.0),
        10: (4.8, 20.0),
        11: (20.0, -8.0),
    }

    def __init__(self):
        super().__init__("cmd_switch")

        self.declare_parameter("initial_stage", 0)
        self.declare_parameter("min_stage_travel_distance", 1.5)
        self.declare_parameter("min_stage_interval_seconds", 2.0)
        self.declare_parameter("use_stage_geofence", True)
        self.declare_parameter("stage_geofence_radius", 3.0)
        self.declare_parameter("use_odom_fallback", True)
        self.declare_parameter("odom_fallback_radius", 2.5)
        self.declare_parameter("final_stage", 10)

        self.active_stage = int(self.get_parameter("initial_stage").value)
        self.min_stage_travel_distance = float(
            self.get_parameter("min_stage_travel_distance").value
        )
        self.min_stage_interval_seconds = float(
            self.get_parameter("min_stage_interval_seconds").value
        )
        self.use_stage_geofence = bool(
            self.get_parameter("use_stage_geofence").value
        )
        self.stage_geofence_radius = float(
            self.get_parameter("stage_geofence_radius").value
        )
        self.use_odom_fallback = bool(
            self.get_parameter("use_odom_fallback").value
        )
        self.odom_fallback_radius = float(
            self.get_parameter("odom_fallback_radius").value
        )
        self.final_stage = int(self.get_parameter("final_stage").value)

        self.released_stage = None

        self.odom_x = None
        self.odom_y = None
        self.odom_yaw = 0.0

        self.last_stage_change_time = None
        self.last_stage_change_x = None
        self.last_stage_change_y = None

        self.detected_signs = {}

        self.stage_topics = {
            5: "/cone_avoid/cmd_vel",
            6: "/dynamic_obstacle/cmd_vel",
            8: "/ramp/cmd_vel",
        }

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/rover/cmd_vel",
            10,
        )
        self.stage_pub = self.create_publisher(
            Int32,
            "/teknofest/stage_id",
            10,
        )
        self.stage_order_pub = self.create_publisher(
            Int32,
            "/teknofest/stage_order",
            10,
        )

        # Subscriptions for cmd_vel sources
        self.fallow_sub = self.create_subscription(
            Twist,
            "/fallow_corridor/cmd_vel",
            self.fallow_callback,
            10,
        )
        self.cone_sub = self.create_subscription(
            Twist,
            "/cone_avoid/cmd_vel",
            self.cone_callback,
            10,
        )
        self.dynamic_sub = self.create_subscription(
            Twist,
            "/dynamic_obstacle/cmd_vel",
            self.dynamic_callback,
            10,
        )
        self.ramp_sub = self.create_subscription(
            Twist,
            "/ramp/cmd_vel",
            self.ramp_callback,
            10,
        )

        # Subscriptions for stage control
        self.release_sub = self.create_subscription(
            Int32,
            "/teknofest/release",
            self.release_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            "/rover/odom",
            self.odom_callback,
            10,
        )
        self.sign_detected_sub = self.create_subscription(
            String,
            "/teknofest/sign_detected",
            self.sign_detected_callback,
            10,
        )

        self.stage_timer = self.create_timer(
            0.2,
            self.publish_stage,
        )

        active_topic = self.stage_topics.get(
            self.active_stage,
            "/fallow_corridor/cmd_vel",
        )
        self.get_logger().info(
            "=== Cmd Switch & Stage Manager Initialized ==="
        )
        self.get_logger().info(
            f"Başlangıç stage: {self.active_stage}"
        )
        self.get_logger().info(
            f"Başlangıç aktif kaynak: {active_topic}"
        )

    def now_seconds(self):
        return (
            self.get_clock()
            .now()
            .nanoseconds
            / 1_000_000_000.0
        )

    def publish_stage(self):
        stage_msg = Int32()
        stage_msg.data = int(self.active_stage)
        self.stage_pub.publish(stage_msg)

        order_msg = Int32()
        order_msg.data = int(self.active_stage)
        self.stage_order_pub.publish(order_msg)

    def record_stage_change(self):
        self.last_stage_change_time = self.now_seconds()
        if self.odom_x is not None and self.odom_y is not None:
            self.last_stage_change_x = self.odom_x
            self.last_stage_change_y = self.odom_y

    def stage_travel_distance(self):
        if (
            self.odom_x is None
            or self.odom_y is None
            or self.last_stage_change_x is None
            or self.last_stage_change_y is None
        ):
            return None
        return math.hypot(
            self.odom_x - self.last_stage_change_x,
            self.odom_y - self.last_stage_change_y,
        )

    def stage_interval_seconds(self):
        if self.last_stage_change_time is None:
            return None
        return self.now_seconds() - self.last_stage_change_time

    def distance_to_expected_stage(self, stage_id):
        try:
            target = self.STAGE_POSITIONS.get(int(stage_id))
        except (ValueError, TypeError):
            target = None

        if target is None or self.odom_x is None or self.odom_y is None:
            return None

        target_x, target_y = target
        return math.hypot(
            self.odom_x - target_x,
            self.odom_y - target_y,
        )

    def stage_change_allowed(self, target_stage=None):
        if target_stage is None:
            target_stage = min(self.active_stage + 1, 11)

        distance = self.stage_travel_distance()
        interval = self.stage_interval_seconds()
        target_distance = self.distance_to_expected_stage(target_stage)

        distance_ok = (
            self.active_stage == 0
            or distance is None
            or distance >= self.min_stage_travel_distance
        )
        interval_ok = (
            self.active_stage == 0
            or interval is None
            or interval >= self.min_stage_interval_seconds
        )
        geofence_ok = (
            not self.use_stage_geofence
            or target_distance is None
            or target_distance <= self.stage_geofence_radius
        )

        return distance_ok and interval_ok and geofence_ok

    def odom_callback(self, msg: Odometry):
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)
        self.odom_yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        if self.last_stage_change_x is None:
            self.last_stage_change_x = self.odom_x
            self.last_stage_change_y = self.odom_y

        self.check_stage_progression()

    def check_stage_progression(self):
        if self.odom_x is None or self.odom_y is None:
            return

        if self.active_stage >= self.final_stage:
            return

        interval = self.stage_interval_seconds()
        if (
            interval is not None
            and self.active_stage > 0
            and interval < self.min_stage_interval_seconds
        ):
            return

        travel_dist = self.stage_travel_distance()
        if (
            travel_dist is not None
            and self.active_stage > 0
            and travel_dist < self.min_stage_travel_distance
        ):
            return

        next_stage = self.active_stage + 1

        target_x, target_y = None, None
        source_desc = ""

        if next_stage in self.detected_signs:
            sign_info = self.detected_signs[next_stage]
            target_x = sign_info["x"]
            target_y = sign_info["y"]
            source_desc = f"Algılanan Tabela ({sign_info.get('label', '')})"
        elif next_stage in self.STAGE_POSITIONS:
            target_x, target_y = self.STAGE_POSITIONS[next_stage]
            source_desc = "Varsayılan Stage Konumu"

        if target_x is not None and target_y is not None:
            dist = math.hypot(
                self.odom_x - target_x,
                self.odom_y - target_y,
            )
            threshold = self.odom_fallback_radius

            if dist <= threshold:
                old_stage = self.active_stage
                self.active_stage = min(self.final_stage, next_stage)
                self.released_stage = None
                self.record_stage_change()
                self.get_logger().info(
                    f"[STAGE GEÇİŞİ] Araç Stage {self.active_stage} konumuna ulaştı/geçti ({source_desc}): "
                    f"Stage {old_stage} -> {self.active_stage} (Konum: ({target_x:.2f}, {target_y:.2f}), Mesafe: {dist:.2f}m <= {threshold:.2f}m)"
                )
                self.publish_stage()

    def sign_detected_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().error(
                f"Sign detected verisi ayrıştırılamadı: {exc}"
            )
            return

        stage_num = data.get("stage_num")
        is_stop = data.get("is_stop", False)
        sign_key = data.get("sign_key")
        x_world = data.get("x")
        y_world = data.get("y")
        label = data.get("label", "sign")

        if sign_key is not None and x_world is not None and y_world is not None:
            key = (
                stage_num
                if stage_num is not None
                else ("STOP" if is_stop else sign_key)
            )
            is_new = key not in self.detected_signs
            self.detected_signs[key] = {
                "x": x_world,
                "y": y_world,
                "label": label,
                "is_stop": is_stop,
            }

            if is_new:
                tag_desc = f"Stage {stage_num}" if stage_num is not None else f"{key}"
                self.get_logger().info(
                    f"[TABELA KONUMU KAYDEDİLDİ] {tag_desc} ({label}) algılandı ve konumu "
                    f"({x_world:.2f}, {y_world:.2f}) olarak haritaya işlendi. "
                    "Araç tabelanın yanına ulaşınca Stage geçişi yapılacak."
                )

    def fallow_callback(self, msg: Twist):
        if self.active_stage not in self.stage_topics:
            self.cmd_vel_pub.publish(msg)

    def cone_callback(self, msg: Twist):
        if self.active_stage == 5:
            self.cmd_vel_pub.publish(msg)

    def dynamic_callback(self, msg: Twist):
        if self.active_stage == 6:
            self.cmd_vel_pub.publish(msg)

    def ramp_callback(self, msg: Twist):
        if self.active_stage == 8:
            self.cmd_vel_pub.publish(msg)

    def release_callback(self, msg: Int32):
        released_stage = int(msg.data)

        if released_stage != self.active_stage:
            return

        self.get_logger().info(
            f"Stage {released_stage} kontrolü bıraktı. "
            "Fallow corridor'a dönülüyor."
        )

        self.released_stage = released_stage
        self.active_stage = 0

        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

        self.publish_stage()


def main(args=None):
    rclpy.init(args=args)
    node = CmdSwitchNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            node.cmd_vel_pub.publish(Twist())

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
