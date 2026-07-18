#!/usr/bin/env python3
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2, Imu
from sensor_msgs_py import point_cloud2 as pc2

class FallowCorridorNode(Node):
    """
    ROS 2 Node to follow a corridor bounded by continuous walls/barriers using 3D Lidar point cloud.
    Uses local grid-based height-difference filtering to ignore terrain slopes and inclines.
    Uses IMU pitch values to scale linear command speed when climbing/descending hills.
    Note: Open-loop control is active. Output command values represent raw motor power factors.
    """
    def __init__(self):
        super().__init__('fallow_corridor')

        # Declare parameters for tuning on the fly (open-loop motor coefficients)
        self.declare_parameter('target_speed', 0.2)           # Target forward speed (motor power: 0.0 to 1.0)
        self.declare_parameter('min_speed', 0.05)              # Minimum forward speed (motor power: 0.0 to 1.0)
        self.declare_parameter('max_angular_speed', 0.4)       # Maximum steering command limit (motor power limit)
        self.declare_parameter('kp_center', 0.8)               # Proportional gain for centering (error -> steering power)
        self.declare_parameter('kp_avoid', 1.0)                # Proportional gain for obstacle avoidance
        
        # Pitch speed scaling parameters
        self.declare_parameter('pitch_scale_factor', 1.5)      # Speed scaling multiplier per pitch radian
        self.declare_parameter('min_speed_factor', 0.5)        # Minimum allowed speed multiplier
        self.declare_parameter('max_speed_factor', 2.0)        # Maximum allowed speed multiplier

        # Crop bounds (spatial box for region of interest)
        self.declare_parameter('lookahead_min_x', 0.3)         # Min distance ahead to consider points (meters)
        self.declare_parameter('lookahead_max_x', 3.5)         # Max distance ahead to consider points (meters)
        self.declare_parameter('lookahead_y', 2.2)             # Max lateral distance to consider walls (meters)
        self.declare_parameter('crop_min_z', -0.65)            # Exclude points below base (lidar frame)
        self.declare_parameter('crop_max_z', 0.65)             # Exclude points above barrier height (lidar frame)

        # Incline filter parameters (local terrain profile segmentation)
        self.declare_parameter('grid_cell_size', 0.2)          # Grid size for local ground estimation (meters)
        self.declare_parameter('min_height_diff', 0.18)        # Height threshold above ground to count as wall (meters)

        # Obstacle parameters
        self.declare_parameter('front_obstacle_dist', 1.8)     # Distance to start slowing down for front obstacles
        self.declare_parameter('front_stop_dist', 0.65)        # Distance to stop the robot completely
        self.declare_parameter('min_points_threshold', 15)     # Min points to consider a wall detected

        # Node State Variables
        self.current_pitch = 0.0
        self.has_imu = False

        # Subscriptions & Publishers
        self.points_sub = self.create_subscription(
            PointCloud2,
            '/rover/points',
            self.pointcloud_callback,
            10
        )
        self.imu_sub = self.create_subscription(
            Imu,
            '/rover/imu',
            self.imu_callback,
            10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/fallow_corridor/cmd_vel',
            10
        )

        self.get_logger().info("=== Fallow Corridor Node Initialized ===")
        self.get_logger().info("Subscribed to /rover/points and /rover/imu")
        self.get_logger().info("Publishing to /fallow_corridor/cmd_vel")

    def imu_callback(self, msg: Imu):
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w
        
        # Calculate pitch angle (theta) from quaternion
        # Pitch is rotation around the transverse/Y axis
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            self.current_pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            self.current_pitch = np.arcsin(sinp)
            
        self.has_imu = True

    def pointcloud_callback(self, msg: PointCloud2):
        # Read parameters
        target_speed = self.get_parameter('target_speed').value
        min_speed = self.get_parameter('min_speed').value
        max_angular_speed = self.get_parameter('max_angular_speed').value
        kp_center = self.get_parameter('kp_center').value
        kp_avoid = self.get_parameter('kp_avoid').value
        pitch_scale_factor = self.get_parameter('pitch_scale_factor').value
        min_speed_factor = self.get_parameter('min_speed_factor').value
        max_speed_factor = self.get_parameter('max_speed_factor').value
        lookahead_min_x = self.get_parameter('lookahead_min_x').value
        lookahead_max_x = self.get_parameter('lookahead_max_x').value
        lookahead_y = self.get_parameter('lookahead_y').value
        crop_min_z = self.get_parameter('crop_min_z').value
        crop_max_z = self.get_parameter('crop_max_z').value
        grid_cell_size = self.get_parameter('grid_cell_size').value
        min_height_diff = self.get_parameter('min_height_diff').value
        front_obstacle_dist = self.get_parameter('front_obstacle_dist').value
        front_stop_dist = self.get_parameter('front_stop_dist').value
        min_points_threshold = self.get_parameter('min_points_threshold').value

        # Unpack PointCloud2 data
        try:
            points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        except Exception as e:
            self.get_logger().error(f"Error unpacking PointCloud2: {str(e)}")
            return

        # 1. Spatial crop to lookahead active volume
        cropped_points = []
        if points:
            for x, y, z in points:
                if (lookahead_min_x <= x <= lookahead_max_x and 
                    -lookahead_y <= y <= lookahead_y and 
                    crop_min_z <= z <= crop_max_z):
                    cropped_points.append((x, y, z))
        else:
            self.get_logger().warn("Empty point cloud received.", throttle_duration_sec=2.0)

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

        left_pts = []
        right_pts = []
        front_pts = []

        # 3. Categorize the segmented vertical obstacle/wall points
        for x, y, z in obstacle_pts:
            # Left wall points (y > 0.25 to exclude front center)
            if y >= 0.25:
                left_pts.append((x, y, z))
            # Right wall points (y < -0.25 to exclude front center)
            elif y <= -0.25:
                right_pts.append((x, y, z))
            
            # Front obstacle sector (directly in front, narrower width)
            if abs(y) <= 0.45 and x <= front_obstacle_dist:
                front_pts.append((x, y, z))

        # Sort and average the closest points to find wall boundaries
        # For left wall, smaller y means closer to robot
        if len(left_pts) >= min_points_threshold:
            left_pts.sort(key=lambda p: p[1])
            left_y = np.mean([p[1] for p in left_pts[:max(1, len(left_pts) // 5)]])
            has_left = True
        else:
            left_y = 1.5  # default nominal value
            has_left = False

        # For right wall, larger y (closest to 0) means closer to robot
        if len(right_pts) >= min_points_threshold:
            right_pts.sort(key=lambda p: p[1], reverse=True)
            right_y = np.mean([p[1] for p in right_pts[:max(1, len(right_pts) // 5)]])
            has_right = True
        else:
            right_y = -1.5  # default nominal value
            has_right = False

        # Calculate steering commands using P-controller
        angular_z = 0.0
        control_mode = "NONE"

        if has_left and has_right:
            # Centering: target is midway between left and right walls
            target_y = (left_y + right_y) / 2.0
            angular_z = kp_center * target_y
            control_mode = "CENTERING"
        elif has_left:
            # Single-wall following (left): maintain 1.5m distance
            error = 1.5 - left_y
            angular_z = -kp_center * error
            control_mode = "FOLLOW_LEFT"
        elif has_right:
            # Single-wall following (right): maintain 1.5m distance
            error = 1.5 - abs(right_y)
            angular_z = kp_center * error
            control_mode = "FOLLOW_RIGHT"
        else:
            control_mode = "NO_WALLS"

        # Handle front obstacle and velocity scaling
        linear_x = target_speed
        
        if front_pts:
            # Find the closest point in front
            closest_front_x = min([p[0] for p in front_pts])
            
            self.get_logger().info(f"Front obstacle detected at {closest_front_x:.2f}m", throttle_duration_sec=1.0)
            
            if closest_front_x <= front_stop_dist:
                # Too close, emergency stop / hard turn away from obstacle centroid
                linear_x = 0.0
                mean_front_y = np.mean([p[1] for p in front_pts])
                angular_z = -kp_avoid * mean_front_y
                control_mode = "AVOID_STOP"
            else:
                # Decelerate proportionally as we get closer to the obstacle
                scale = (closest_front_x - front_stop_dist) / (front_obstacle_dist - front_stop_dist)
                linear_x = max(min_speed, target_speed * scale)
                # Assist steering to avoid obstacle
                mean_front_y = np.mean([p[1] for p in front_pts])
                angular_z += -kp_avoid * mean_front_y * (1.0 - scale)
                control_mode += "+AVOID"

        # 4. Incline Speed Compensation
        # Negative pitch means nose is tilted up (positive slope -> increase power)
        # Positive pitch means nose is tilted down (negative slope -> decrease power)
        speed_factor = 1.0
        if self.has_imu and linear_x > 0.0:
            speed_factor = 1.0 - (self.current_pitch * pitch_scale_factor)
            speed_factor = np.clip(speed_factor, min_speed_factor, max_speed_factor)
            linear_x = linear_x * speed_factor

        # Clamp angular velocity to prevent violent open-loop spinning
        angular_z = np.clip(angular_z, -max_angular_speed, max_angular_speed)

        # Publish velocities
        cmd_msg = Twist()
        cmd_msg.linear.x = float(linear_x)
        cmd_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(cmd_msg)

        # Log status
        left_str = f"{left_y:.2f}m" if has_left else "N/A"
        right_str = f"{right_y:.2f}m" if has_right else "N/A"
        self.get_logger().info(
            f"[{control_mode}] L_wall: {left_str} | R_wall: {right_str} | "
            f"Pitch: {self.current_pitch:.3f} rad | "
            f"Cmd Power: v={linear_x:.3f} (factor={speed_factor:.2f}), w={angular_z:.3f}",
            throttle_duration_sec=0.5
        )

    def stop_robot(self):
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FallowCorridorNode()
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
