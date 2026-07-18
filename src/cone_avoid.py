#!/usr/bin/env python3

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Int32


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class ConeAvoidNode(Node):
    """
    Stage 5 trafik konisi parkuru için PointCloud2 tabanlı
    koniden kaçınma node'u.

    Koordinatlar:
        x > 0  : robotun önü
        y > 0  : robotun solu
        y < 0  : robotun sağı

    Stage çalışma mantığı:
        Stage 5  -> cone_avoid /rover/cmd_vel yayınlar.
        Stage 6  -> cone_avoid yayın yapmayı bırakır.
    """

    LEFT = 1
    RIGHT = -1

    def __init__(self):
        super().__init__("cone_avoid")

        # Stage kontrolü
        self.declare_parameter("initial_stage", 5)

        # Hareket parametreleri
        self.declare_parameter("target_speed", 0.18)
        self.declare_parameter("slow_speed", 0.07)
        self.declare_parameter("kp_lateral", 1.30)
        self.declare_parameter("kp_heading", 0.55)
        self.declare_parameter("max_angular_speed", 0.50)
        self.declare_parameter("hard_turn_speed", 0.50)
        self.declare_parameter("smoothing_alpha", 0.45)

        # PointCloud çalışma alanı
        self.declare_parameter("lookahead_min_x", 0.15)
        self.declare_parameter("lookahead_max_x", 3.00)
        self.declare_parameter("crop_half_width", 1.80)
        self.declare_parameter("crop_min_z", -0.65)
        self.declare_parameter("crop_max_z", 0.75)

        # Zemin ayıklama
        self.declare_parameter("ground_grid_size", 0.20)
        self.declare_parameter("min_height_diff", 0.10)

        # Koni kümeleme
        self.declare_parameter("cluster_cell_size", 0.18)
        self.declare_parameter("cone_min_points", 5)
        self.declare_parameter("cone_max_width", 0.95)
        self.declare_parameter("cone_max_depth", 0.95)

        # Koni algılama ve yol seçimi
        self.declare_parameter("cone_detect_distance", 2.50)
        self.declare_parameter("cone_search_half_width", 1.40)
        self.declare_parameter("pair_max_x_gap", 1.25)
        self.declare_parameter("safe_lateral_clearance", 0.80)
        self.declare_parameter("collision_half_width", 0.62)
        self.declare_parameter("max_target_offset", 0.65)
        self.declare_parameter("danger_distance", 1.00)
        self.declare_parameter("default_avoid_side", -1)

        # Acil çarpışma koruması
        self.declare_parameter("emergency_distance", 0.65)
        self.declare_parameter("emergency_half_width", 0.62)
        self.declare_parameter("emergency_min_points", 3)

        self.current_stage = int(self.get_parameter("initial_stage").value)

        self.current_yaw = 0.0
        self.initial_yaw = None
        self.has_imu = False

        self.previous_angular = 0.0
        self.last_steer_sign = self.RIGHT

        self.points_sub = self.create_subscription(
            PointCloud2,
            "/rover/points",
            self.pointcloud_callback,
            qos_profile_sensor_data
        )

        self.imu_sub = self.create_subscription(
            Imu,
            "/rover/imu",
            self.imu_callback,
            qos_profile_sensor_data
        )

        self.stage_sub = self.create_subscription(
            Int32,
            "/teknofest/stage_id",
            self.stage_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/cone_avoid/cmd_vel",
            10
        )

        self.get_logger().info("=== Cone Avoid Node başladı ===")
        self.get_logger().info(f"Başlangıç stage: {self.current_stage}")
        self.get_logger().info("Abone: /rover/points")
        self.get_logger().info("Abone: /rover/imu")
        self.get_logger().info("Abone: /teknofest/stage_id")
        self.get_logger().info("Yayın: /cone_avoid/cmd_vel")

        if self.current_stage == 5:
            self.get_logger().info("Stage 5 aktif: cone_avoid kontrolü aldı.")
        else:
            self.get_logger().info("Stage 5 aktif değil: cone_avoid beklemede.")

    def stage_callback(self, msg):
        detected_stage = int(msg.data)

        if detected_stage <= 0:
            return

        if detected_stage == self.current_stage:
            return

        # Stage numarasının yanlış algılanıp ileri atlamasını engeller.
        # Cone avoid yalnızca Stage 5'e geçişi (aktivasyon) ve Stage 6'ya geçişi (deaktivasyon) bekler.
        if self.current_stage < 5:
            if detected_stage != 5:
                return
        elif self.current_stage == 5:
            if detected_stage != 6:
                return

        old_stage = self.current_stage
        self.current_stage = detected_stage

        self.get_logger().info(
            f"Stage geçişi kabul edildi: {old_stage} -> {self.current_stage}"
        )

        if old_stage == 5 and self.current_stage != 5:
            # Son gönderilen hareket komutunun Gazebo'da kalmaması için
            # yalnızca bir defa dur komutu yollar.
            self.publish_command(0.0, 0.0)

            self.previous_angular = 0.0

            self.get_logger().info(
                "Cone avoid kontrolü bırakıldı. "
                "Stage 6 kontrolü fallow_corridor node'una geçebilir."
            )

        elif self.current_stage == 5:
            self.initial_yaw = self.current_yaw if self.has_imu else None
            self.previous_angular = 0.0

            self.get_logger().info(
                "Stage 5 aktif oldu. Cone avoid kontrolü devraldı."
            )

    def imu_callback(self, msg):
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)

        self.current_yaw = math.atan2(sin_yaw, cos_yaw)
        self.has_imu = True

        if self.initial_yaw is None and self.current_stage == 5:
            self.initial_yaw = self.current_yaw

    def extract_obstacle_points(self, msg):
        lookahead_min_x = float(self.get_parameter("lookahead_min_x").value)
        lookahead_max_x = float(self.get_parameter("lookahead_max_x").value)
        crop_half_width = float(self.get_parameter("crop_half_width").value)
        crop_min_z = float(self.get_parameter("crop_min_z").value)
        crop_max_z = float(self.get_parameter("crop_max_z").value)
        ground_grid_size = float(self.get_parameter("ground_grid_size").value)
        min_height_diff = float(self.get_parameter("min_height_diff").value)

        try:
            raw_points = pc2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True
            )
        except Exception as exc:
            self.get_logger().error(f"PointCloud okunamadı: {exc}")
            return []

        cropped_points = []

        for point in raw_points:
            x = float(point[0])
            y = float(point[1])
            z = float(point[2])

            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue

            if not (lookahead_min_x <= x <= lookahead_max_x):
                continue

            if abs(y) > crop_half_width:
                continue

            if not (crop_min_z <= z <= crop_max_z):
                continue

            cropped_points.append((x, y, z))

        if not cropped_points:
            return []

        cells = {}

        for x, y, z in cropped_points:
            cell_x = math.floor(x / ground_grid_size)
            cell_y = math.floor(y / ground_grid_size)
            cell_id = (cell_x, cell_y)

            if cell_id not in cells:
                cells[cell_id] = []

            cells[cell_id].append((x, y, z))

        obstacle_points = []

        for cell_points in cells.values():
            ground_z = min(point[2] for point in cell_points)

            for point in cell_points:
                height_difference = point[2] - ground_z

                if height_difference >= min_height_diff:
                    obstacle_points.append(point)

        return obstacle_points

    def cluster_obstacles(self, obstacle_points):
        if not obstacle_points:
            return []

        cell_size = float(self.get_parameter("cluster_cell_size").value)

        cell_map = {}

        for point in obstacle_points:
            x, y, _ = point

            cell_x = math.floor(x / cell_size)
            cell_y = math.floor(y / cell_size)
            cell_id = (cell_x, cell_y)

            if cell_id not in cell_map:
                cell_map[cell_id] = []

            cell_map[cell_id].append(point)

        unvisited = set(cell_map.keys())
        clusters = []

        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        while unvisited:
            start_cell = unvisited.pop()
            stack = [start_cell]
            component_cells = [start_cell]

            while stack:
                current_x, current_y = stack.pop()

                for offset_x, offset_y in neighbors:
                    neighbor = (
                        current_x + offset_x,
                        current_y + offset_y
                    )

                    if neighbor not in unvisited:
                        continue

                    unvisited.remove(neighbor)
                    stack.append(neighbor)
                    component_cells.append(neighbor)

            component_points = []

            for cell_id in component_cells:
                component_points.extend(cell_map[cell_id])

            if not component_points:
                continue

            x_values = [point[0] for point in component_points]
            y_values = [point[1] for point in component_points]
            z_values = [point[2] for point in component_points]

            cluster = {
                "count": len(component_points),
                "x_min": min(x_values),
                "x_max": max(x_values),
                "y_min": min(y_values),
                "y_max": max(y_values),
                "z_min": min(z_values),
                "z_max": max(z_values),
                "center_x": float(np.mean(x_values)),
                "center_y": float(np.mean(y_values))
            }

            cluster["depth"] = cluster["x_max"] - cluster["x_min"]
            cluster["width"] = cluster["y_max"] - cluster["y_min"]

            clusters.append(cluster)

        return clusters

    def detect_cones(self, obstacle_points):
        detect_distance = float(self.get_parameter("cone_detect_distance").value)
        search_half_width = float(self.get_parameter("cone_search_half_width").value)
        min_points = int(self.get_parameter("cone_min_points").value)
        max_width = float(self.get_parameter("cone_max_width").value)
        max_depth = float(self.get_parameter("cone_max_depth").value)

        clusters = self.cluster_obstacles(obstacle_points)
        candidates = []

        for cluster in clusters:
            if cluster["count"] < min_points:
                continue

            if cluster["x_min"] > detect_distance:
                continue

            if abs(cluster["center_y"]) > search_half_width:
                continue

            if cluster["width"] > max_width:
                continue

            if cluster["depth"] > max_depth:
                continue

            candidates.append(cluster)

        candidates.sort(
            key=lambda item: (
                item["x_min"],
                abs(item["center_y"])
            )
        )

        return candidates

    def choose_default_side(self):
        default_side = int(self.get_parameter("default_avoid_side").value)

        if default_side >= 0:
            return self.LEFT

        return self.RIGHT

    def calculate_target_path(self, cones):
        if not cones:
            return 0.0, None, "CRUISE", []

        pair_max_x_gap = float(self.get_parameter("pair_max_x_gap").value)
        safe_clearance = float(self.get_parameter("safe_lateral_clearance").value)
        collision_half_width = float(self.get_parameter("collision_half_width").value)
        max_target_offset = float(self.get_parameter("max_target_offset").value)

        left_cones = [cone for cone in cones if cone["center_y"] >= 0.0]
        right_cones = [cone for cone in cones if cone["center_y"] < 0.0]

        nearest_left = None
        nearest_right = None

        if left_cones:
            nearest_left = min(left_cones, key=lambda cone: cone["x_min"])

        if right_cones:
            nearest_right = min(right_cones, key=lambda cone: cone["x_min"])

        # Sağda ve solda birbirine yakın iki koni varsa
        # ikisinin arasındaki orta noktaya yönel.
        if nearest_left is not None and nearest_right is not None:
            x_gap = abs(nearest_left["x_min"] - nearest_right["x_min"])

            if x_gap <= pair_max_x_gap:
                target_y = (
                    nearest_left["center_y"]
                    + nearest_right["center_y"]
                ) / 2.0

                target_y = float(
                    np.clip(
                        target_y,
                        -max_target_offset,
                        max_target_offset
                    )
                )

                nearest_x = min(
                    nearest_left["x_min"],
                    nearest_right["x_min"]
                )

                return (
                    target_y,
                    nearest_x,
                    "BETWEEN_CONES",
                    [nearest_left, nearest_right]
                )

        nearest_cone = cones[0]
        cone_y = nearest_cone["center_y"]

        # Koni aracın çarpışma koridorunun yeterince dışındaysa
        # gereksiz sert manevra yapma.
        if abs(cone_y) > collision_half_width:
            return (
                0.0,
                nearest_cone["x_min"],
                "SIDE_CONE",
                [nearest_cone]
            )

        # Sol taraftaki koninin sağından geç.
        if cone_y > 0.08:
            target_y = cone_y - safe_clearance
            mode = "AVOID_LEFT_CONE"

        # Sağ taraftaki koninin solundan geç.
        elif cone_y < -0.08:
            target_y = cone_y + safe_clearance
            mode = "AVOID_RIGHT_CONE"

        # Tam ortadaki koni için varsayılan kaçış yönünü kullan.
        else:
            side = self.choose_default_side()
            target_y = side * safe_clearance
            mode = "AVOID_CENTER_CONE"

        target_y = float(
            np.clip(
                target_y,
                -max_target_offset,
                max_target_offset
            )
        )

        return (
            target_y,
            nearest_cone["x_min"],
            mode,
            [nearest_cone]
        )

    def find_emergency_obstacle(self, obstacle_points):
        emergency_distance = float(self.get_parameter("emergency_distance").value)
        emergency_half_width = float(self.get_parameter("emergency_half_width").value)
        emergency_min_points = int(self.get_parameter("emergency_min_points").value)

        emergency_points = [
            point
            for point in obstacle_points
            if point[0] <= emergency_distance
            and abs(point[1]) <= emergency_half_width
        ]

        if len(emergency_points) < emergency_min_points:
            return None

        return {
            "x_min": min(point[0] for point in emergency_points),
            "center_y": float(np.mean([point[1] for point in emergency_points])),
            "count": len(emergency_points)
        }

    def calculate_speed(self, nearest_x, mode):
        target_speed = float(self.get_parameter("target_speed").value)
        slow_speed = float(self.get_parameter("slow_speed").value)
        detect_distance = float(self.get_parameter("cone_detect_distance").value)
        danger_distance = float(self.get_parameter("danger_distance").value)

        if nearest_x is None:
            return target_speed

        if mode in ("CRUISE", "SIDE_CONE"):
            return target_speed

        if nearest_x <= danger_distance:
            return slow_speed

        denominator = max(0.01, detect_distance - danger_distance)
        scale = (nearest_x - danger_distance) / denominator
        scale = float(np.clip(scale, 0.0, 1.0))

        return slow_speed + (target_speed - slow_speed) * scale

    def calculate_steering(self, target_y, nearest_x, mode):
        kp_lateral = float(self.get_parameter("kp_lateral").value)
        kp_heading = float(self.get_parameter("kp_heading").value)
        max_angular = float(self.get_parameter("max_angular_speed").value)
        smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        detect_distance = float(self.get_parameter("cone_detect_distance").value)
        danger_distance = float(self.get_parameter("danger_distance").value)

        heading_error = 0.0

        if self.has_imu and self.initial_yaw is not None:
            heading_error = normalize_angle(
                self.initial_yaw - self.current_yaw
            )

        if nearest_x is None or mode in ("CRUISE", "SIDE_CONE"):
            raw_angular = kp_heading * heading_error

        else:
            denominator = max(0.01, detect_distance - danger_distance)
            urgency = (detect_distance - nearest_x) / denominator
            urgency = float(np.clip(urgency, 0.0, 1.0))

            lateral_gain = 0.55 + 0.75 * urgency

            raw_angular = kp_lateral * target_y * lateral_gain
            raw_angular += 0.15 * kp_heading * heading_error

        raw_angular = float(
            np.clip(
                raw_angular,
                -max_angular,
                max_angular
            )
        )

        angular_z = (
            (1.0 - smoothing_alpha) * self.previous_angular
            + smoothing_alpha * raw_angular
        )

        angular_z = float(
            np.clip(
                angular_z,
                -max_angular,
                max_angular
            )
        )

        self.previous_angular = angular_z

        if abs(angular_z) > 0.05:
            self.last_steer_sign = self.LEFT if angular_z > 0.0 else self.RIGHT

        return angular_z

    def emergency_turn_direction(self, emergency, target_y):
        # Engel soldaysa sağa dön.
        if emergency["center_y"] > 0.05:
            return self.RIGHT

        # Engel sağdaysa sola dön.
        if emergency["center_y"] < -0.05:
            return self.LEFT

        # Engel ortadaysa hesaplanan hedef yönünü kullan.
        if target_y > 0.05:
            return self.LEFT

        if target_y < -0.05:
            return self.RIGHT

        if self.last_steer_sign in (self.LEFT, self.RIGHT):
            return self.last_steer_sign

        return self.choose_default_side()

    def publish_command(self, linear_x, angular_z):
        max_angular = float(self.get_parameter("max_angular_speed").value)

        command = Twist()
        command.linear.x = float(linear_x)
        command.angular.z = float(np.clip(angular_z, -max_angular, max_angular))

        self.cmd_vel_pub.publish(command)

    def pointcloud_callback(self, msg):
        # Stage 5 dışında cone_avoid hız komutu yayınlamaz.
        if self.current_stage != 5:
            return

        obstacle_points = self.extract_obstacle_points(msg)
        cones = self.detect_cones(obstacle_points)

        target_y, nearest_x, mode, relevant_cones = self.calculate_target_path(
            cones
        )

        emergency = self.find_emergency_obstacle(obstacle_points)

        if emergency is not None:
            turn_side = self.emergency_turn_direction(
                emergency,
                target_y
            )

            hard_turn_speed = float(
                self.get_parameter("hard_turn_speed").value
            )

            linear_x = 0.0
            angular_z = turn_side * hard_turn_speed

            self.previous_angular = angular_z
            self.last_steer_sign = turn_side
            mode = "EMERGENCY_TURN"

            direction_text = "LEFT" if turn_side == self.LEFT else "RIGHT"

            self.get_logger().warning(
                f"ACİL ENGEL: "
                f"x={emergency['x_min']:.2f}m, "
                f"y={emergency['center_y']:.2f}m, "
                f"yön={direction_text}",
                throttle_duration_sec=0.5
            )

        else:
            linear_x = self.calculate_speed(nearest_x, mode)
            angular_z = self.calculate_steering(
                target_y,
                nearest_x,
                mode
            )

        self.publish_command(linear_x, angular_z)

        if relevant_cones:
            cone_text = " | ".join(
                f"x={cone['x_min']:.2f}, "
                f"y={cone['center_y']:.2f}, "
                f"pts={cone['count']}"
                for cone in relevant_cones
            )
        else:
            cone_text = "none"

        self.get_logger().info(
            f"[Stage {self.current_stage} | {mode}] "
            f"target_y={target_y:.2f} | "
            f"cones={cone_text} | "
            f"v={linear_x:.2f}, "
            f"w={angular_z:.2f}",
            throttle_duration_sec=0.5
        )

    def stop_robot(self):
        if not rclpy.ok():
            return

        try:
            command = Twist()
            command.linear.x = 0.0
            command.angular.z = 0.0
            self.cmd_vel_pub.publish(command)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ConeAvoidNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.stop_robot()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
