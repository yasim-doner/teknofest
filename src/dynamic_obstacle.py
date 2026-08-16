#!/usr/bin/env python3

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Int32


class DynamicObstacleNode(Node):
    """
    ROS 2 Node for Stage 6 (Dynamic Obstacle / Sliding Barrier).
    Waits in front of the barrier while the corridor is blocked.
    As soon as the sliding barrier opens, moves straight forward for crossing_duration seconds,
    and then releases control back to fallow_corridor.
    """

    def __init__(self):
        super().__init__("dynamic_obstacle")

        # Parameters
        self.declare_parameter("initial_stage", 0)
        self.declare_parameter("target_speed", 0.35)
        self.declare_parameter("brake_speed", -0.15)
        self.declare_parameter("brake_duration", 0.4)
        self.declare_parameter("detection_min_x", 0.4)
        self.declare_parameter("detection_max_x", 2.2)
        self.declare_parameter("detection_y", 0.45)
        self.declare_parameter("crop_min_z", -0.65)
        self.declare_parameter("crop_max_z", 0.65)
        self.declare_parameter("grid_cell_size", 0.2)
        self.declare_parameter("min_height_diff", 0.18)
        self.declare_parameter("min_points_threshold", 15)
        self.declare_parameter("crossing_duration", 3.0)

        initial_stage = int(self.get_parameter("initial_stage").value)
        self.is_active = initial_stage == 6
        self.clear_start_time = None
        self.stop_start_time = None

        # Subscriptions & Publishers
        self.points_sub = self.create_subscription(
            PointCloud2,
            "/rover/points",
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )
        self.stage_sub = self.create_subscription(
            Int32,
            "/teknofest/stage_id",
            self.stage_callback,
            10,
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/dynamic_obstacle/cmd_vel",
            10,
        )
        self.release_pub = self.create_publisher(
            Int32,
            "/teknofest/release",
            10,
        )

        self.get_logger().info("=== Dynamic Obstacle Node (Simplified) Initialized ===")

    def stage_callback(self, msg: Int32):
        if msg.data == 6:
            if not self.is_active:
                self.is_active = True
                self.clear_start_time = None
                self.get_logger().info("Dynamic obstacle node ACTIVATED for Stage 6.")
        else:
            if self.is_active:
                self.is_active = False
                self.clear_start_time = None
                self.get_logger().info("Dynamic obstacle node DEACTIVATED.")

    def pointcloud_callback(self, msg: PointCloud2):
        if not self.is_active:
            return

        target_speed = float(self.get_parameter("target_speed").value)
        brake_speed = float(self.get_parameter("brake_speed").value)
        brake_duration = float(self.get_parameter("brake_duration").value)
        detection_min_x = float(self.get_parameter("detection_min_x").value)
        detection_max_x = float(self.get_parameter("detection_max_x").value)
        detection_y = float(self.get_parameter("detection_y").value)
        crop_min_z = float(self.get_parameter("crop_min_z").value)
        crop_max_z = float(self.get_parameter("crop_max_z").value)
        grid_cell_size = float(self.get_parameter("grid_cell_size").value)
        min_height_diff = float(self.get_parameter("min_height_diff").value)
        min_points_threshold = int(self.get_parameter("min_points_threshold").value)
        crossing_duration = float(self.get_parameter("crossing_duration").value)

        # 1. Active Crossing State
        if self.clear_start_time is not None:
            elapsed = (self.get_clock().now() - self.clear_start_time).nanoseconds / 1e9
            if elapsed < crossing_duration:
                cmd_msg = Twist()
                cmd_msg.linear.x = target_speed
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)
                self.get_logger().info(
                    f"[CROSSING] Süre: {elapsed:.2f}s / {crossing_duration:.2f}s | Hız: {target_speed:.2f} m/s",
                    throttle_duration_sec=0.5,
                )
                return
            else:
                self.get_logger().info("[CROSSING TAMAMLANDI] Geçiş tamamlandı, kontrol devrediliyor.")
                release_msg = Int32()
                release_msg.data = 6
                self.release_pub.publish(release_msg)

                self.stop_robot()
                self.is_active = False
                self.clear_start_time = None
                self.stop_start_time = None
                return

        # 2. Extract PointCloud2
        try:
            pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        except Exception as e:
            self.get_logger().error(f"PointCloud2 okuma hatası: {str(e)}")
            return

        if not pts:
            return

        # Handle potential structured numpy record array
        arr = np.asarray(pts)
        if arr.dtype.names is not None:
            points_np = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float32)
        else:
            points_np = np.array([(p[0], p[1], p[2]) for p in pts], dtype=np.float32)

        # 3. Crop to front central corridor
        mask = (
            (points_np[:, 0] >= detection_min_x)
            & (points_np[:, 0] <= detection_max_x)
            & (points_np[:, 1] >= -detection_y)
            & (points_np[:, 1] <= detection_y)
            & (points_np[:, 2] >= crop_min_z)
            & (points_np[:, 2] <= crop_max_z)
        )
        cropped = points_np[mask]

        # 4. Local terrain height filter (ignore slope/ground floor)
        obstacle_count = 0
        if len(cropped) > 0:
            cells = {}
            for p in cropped:
                cx = int(p[0] // grid_cell_size)
                cy = int(p[1] // grid_cell_size)
                cell_id = (cx, cy)
                if cell_id not in cells:
                    cells[cell_id] = []
                cells[cell_id].append(p[2])

            for cell_id, z_list in cells.items():
                min_z = min(z_list)
                for z_val in z_list:
                    if (z_val - min_z) >= min_height_diff:
                        obstacle_count += 1

        # 5. Barrier State Decision
        if obstacle_count >= min_points_threshold:
            # Barrier CLOSED: Stop & Wait with brief reverse braking pulse
            if self.stop_start_time is None:
                self.stop_start_time = self.get_clock().now()

            elapsed_stop = (self.get_clock().now() - self.stop_start_time).nanoseconds / 1e9
            if elapsed_stop < brake_duration:
                cmd_msg = Twist()
                cmd_msg.linear.x = brake_speed
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)
            else:
                self.stop_robot()

            self.get_logger().info(
                f"[DUR-BEKLE] Bariyer kapalı (Nokta: {obstacle_count}). Ters Fren: {brake_speed} m/s ({elapsed_stop:.2f}s)",
                throttle_duration_sec=0.5,
            )
        else:
            # Barrier OPEN: Start crossing
            self.stop_start_time = None
            self.clear_start_time = self.get_clock().now()
            cmd_msg = Twist()
            cmd_msg.linear.x = target_speed
            cmd_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_msg)
            self.get_logger().info(
                f"[GEÇİŞ BAŞLADI] Ön koridor temiz (Nokta: {obstacle_count} < {min_points_threshold}). İlerleniyor."
            )

    def stop_robot(self):
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
