#!/usr/bin/env python3
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Int32

class DynamicObstacleNode(Node):
    """
    ROS 2 Node to wait until a dynamic obstacle (sliding barrier) in front of the robot passes,
    and then move straight forward with no rotation.
    Note: Open-loop control is active. Output command values represent raw motor power factors.
    """
    def __init__(self):
        super().__init__('dynamic_obstacle')

        # Declare parameters for tuning on the fly
        self.declare_parameter('target_speed', 0.25)           # Target forward speed when path is clear (motor power)
        self.declare_parameter('detection_min_x', 0.4)         # Min distance ahead to detect obstacle (meters)
        self.declare_parameter('detection_max_x', 2.2)         # Max distance ahead to detect obstacle (meters)
        self.declare_parameter('detection_y', 0.6)             # Half-width of detection sector (meters)
        self.declare_parameter('crop_min_z', -0.65)            # Exclude points below base (lidar frame)
        self.declare_parameter('crop_max_z', 0.65)             # Exclude points above barrier height (lidar frame)

        # Incline filter parameters (local terrain profile segmentation)
        self.declare_parameter('grid_cell_size', 0.2)          # Grid size for local ground estimation (meters)
        self.declare_parameter('min_height_diff', 0.18)        # Height threshold above ground to count as obstacle (meters)
        self.declare_parameter('min_points_threshold', 15)     # Min points to consider obstacle present
        self.declare_parameter('crossing_duration', 4.0)       # Seconds to move forward after obstacle clears

        # State variables for activation and passing
        self.is_active = False
        self.clear_start_time = None

        # Subscriptions & Publishers
        self.points_sub = self.create_subscription(
            PointCloud2,
            '/rover/points',
            self.pointcloud_callback,
            10
        )
        self.stage_sub = self.create_subscription(
            Int32,
            '/teknofest/stage_id',
            self.stage_callback,
            10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/dynamic_obstacle/cmd_vel',
            10
        )
        self.release_pub = self.create_publisher(
            Int32,
            '/teknofest/release',
            10
        )

        self.get_logger().info("=== Dynamic Obstacle Node Initialized ===")
        self.get_logger().info("Subscribed to /rover/points and /teknofest/stage_id")
        self.get_logger().info("Publishing to /dynamic_obstacle/cmd_vel and /teknofest/release")

    def stage_callback(self, msg: Int32):
        if msg.data == 6:
            if not self.is_active:
                self.is_active = True
                self.clear_start_time = None
                self.get_logger().info("Dynamic obstacle node ACTIVATED for Stage 6.")

    def pointcloud_callback(self, msg: PointCloud2):
        if not self.is_active:
            return

        # Read parameters
        target_speed = self.get_parameter('target_speed').value
        detection_min_x = self.get_parameter('detection_min_x').value
        detection_max_x = self.get_parameter('detection_max_x').value
        detection_y = self.get_parameter('detection_y').value
        crop_min_z = self.get_parameter('crop_min_z').value
        crop_max_z = self.get_parameter('crop_max_z').value
        grid_cell_size = self.get_parameter('grid_cell_size').value
        min_height_diff = self.get_parameter('min_height_diff').value
        min_points_threshold = self.get_parameter('min_points_threshold').value
        crossing_duration = self.get_parameter('crossing_duration').value

        # If we have already started crossing, drive forward and check timer
        if self.clear_start_time is not None:
            elapsed = (self.get_clock().now() - self.clear_start_time).nanoseconds / 1e9
            if elapsed < crossing_duration:
                cmd_msg = Twist()
                cmd_msg.linear.x = float(target_speed)
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)
                self.get_logger().info(f"[CROSSING] Elapsed: {elapsed:.2f}s / {crossing_duration:.2f}s", throttle_duration_sec=0.5)
                return
            else:
                # Finished crossing! Release control.
                self.get_logger().info("Crossing completed. Releasing control back to fallow_corridor.")
                
                # Publish release message
                release_msg = Int32()
                release_msg.data = 6
                self.release_pub.publish(release_msg)
                
                # Stop robot and reset state
                self.stop_robot()
                self.is_active = False
                self.clear_start_time = None
                return

        # Unpack PointCloud2 data
        try:
            points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        except Exception as e:
            self.get_logger().error(f"Error unpacking PointCloud2: {str(e)}")
            return

        # 1. Spatial crop to the front detection sector
        cropped_points = []
        if points:
            for x, y, z in points:
                if (detection_min_x <= x <= detection_max_x and 
                    -detection_y <= y <= detection_y and 
                    crop_min_z <= z <= crop_max_z):
                    cropped_points.append((x, y, z))

        # 2. Local terrain segmentation (ignore slopes)
        # Group cropped points into 2D grid cells in horizontal XY plane
        cells = {}
        for p in cropped_points:
            cx = int(p[0] // grid_cell_size)
            cy = int(p[1] // grid_cell_size)
            cell_id = (cx, cy)
            if cell_id not in cells:
                cells[cell_id] = []
            cells[cell_id].append(p)

        # Keep only points that are significantly higher than the local ground floor (min Z) in their cell
        obstacle_pts = []
        for cell_id, cell_points in cells.items():
            min_z_val = min([pt[2] for pt in cell_points])
            for pt in cell_points:
                if (pt[2] - min_z_val) >= min_height_diff:
                    obstacle_pts.append(pt)

        # 3. Decision Logic
        obstacle_detected = len(obstacle_pts) >= min_points_threshold

        if obstacle_detected:
            linear_x = 0.0
            status_msg = f"WAITING (Obstacle detected: {len(obstacle_pts)} pts)"
            
            # Publish velocities (stop)
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_msg)
        else:
            # Obstacle cleared, start the crossing timer
            self.clear_start_time = self.get_clock().now()
            linear_x = target_speed
            status_msg = f"MOVING CLEAR - START TIMER (Obstacle points: {len(obstacle_pts)} pts)"
            
            # Publish velocities
            cmd_msg = Twist()
            cmd_msg.linear.x = float(linear_x)
            cmd_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_msg)

        # Log status
        self.get_logger().info(
            f"[{status_msg}] Cmd Power: v={linear_x:.3f}, w=0.000",
            throttle_duration_sec=0.5
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

if __name__ == '__main__':
    main()
