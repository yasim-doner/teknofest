#!/usr/bin/env python3
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
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
        self.declare_parameter('single_wall_target_dist', 1.2)  # Target distance from wall in single-wall mode (meters)

        # Node State Variables
        self.current_pitch = 0.0
        self.has_imu = False
        self.last_angular_z = 0.0
        self.no_walls_count = 0

        # Subscriptions & Publishers
        self.points_sub = self.create_subscription(
            PointCloud2,
            '/rover/points',
            self.pointcloud_callback,
            qos_profile_sensor_data
        )
        self.imu_sub = self.create_subscription(
            Imu,
            '/rover/imu',
            self.imu_callback,
            qos_profile_sensor_data
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
        single_wall_target_dist = self.get_parameter('single_wall_target_dist').value

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

        # 2. Local terrain segmentation (ignore ground slopes)
        cells = {}
        for p in cropped_points:
            cx = int(p[0] // grid_cell_size)
            cy = int(p[1] // grid_cell_size)
            cell_id = (cx, cy)
            if cell_id not in cells:
                cells[cell_id] = []
            cells[cell_id].append(p)

        obstacle_pts = []
        for cell_id, cell_points in cells.items():
            min_z_val = min([pt[2] for pt in cell_points])
            for pt in cell_points:
                if (pt[2] - min_z_val) >= min_height_diff:
                    obstacle_pts.append(pt)

        # 3. U-Corridor Dual-Slice (Near & Far) Path Estimation
        # Split obstacle points into NEAR slice (0.05m - 1.5m) and FAR slice (1.5m - 3.8m)
        near_pts = [p for p in obstacle_pts if 0.05 <= p[0] <= 1.5]
        far_pts  = [p for p in obstacle_pts if 1.5 < p[0] <= lookahead_max_x]

        # --- NEAR SLICE (Local Centering) ---
        near_left  = [p[1] for p in near_pts if p[1] >= 0.20]
        near_right = [p[1] for p in near_pts if p[1] <= -0.20]

        has_near_left  = len(near_left) >= min_points_threshold
        has_near_right = len(near_right) >= min_points_threshold

        if has_near_left and has_near_right:
            y_nl = np.mean(sorted(near_left)[:max(1, len(near_left) // 4)])
            y_nr = np.mean(sorted(near_right, reverse=True)[:max(1, len(near_right) // 4)])
            mid_y_near = (y_nl + y_nr) / 2.0
            mode_near = "BOTH"
        elif has_near_left:
            y_nl = np.mean(sorted(near_left)[:max(1, len(near_left) // 4)])
            mid_y_near = y_nl - 1.15
            mode_near = "LEFT_ONLY"
        elif has_near_right:
            y_nr = np.mean(sorted(near_right, reverse=True)[:max(1, len(near_right) // 4)])
            mid_y_near = y_nr + 1.15
            mode_near = "RIGHT_ONLY"
        else:
            mid_y_near = 0.0
            mode_near = "NONE"

        # --- FAR SLICE (U-Turn / Ahead Corridor Curve Detection) ---
        far_center = [p for p in far_pts if abs(p[1]) <= 0.50]
        far_left   = [p[1] for p in far_pts if p[1] >= 0.20]
        far_right  = [p[1] for p in far_pts if p[1] <= -0.20]

        has_far_center = len(far_center) >= min_points_threshold
        has_far_left   = len(far_left) >= min_points_threshold
        has_far_right  = len(far_right) >= min_points_threshold

        if has_far_center:
            # U-turn curve or wall blocking straight path ahead!
            # Determine open corridor direction
            if len(far_left) > len(far_right) + 3:
                # Outer U-bend wall is on the left -> U-turn goes RIGHT!
                mid_y_far = -1.6
                mode_far = "U_TURN_RIGHT"
            elif len(far_right) > len(far_left) + 3:
                # Outer U-bend wall is on the right -> U-turn goes LEFT!
                mid_y_far = +1.6
                mode_far = "U_TURN_LEFT"
            else:
                # Symmetric wall directly ahead: steer towards open side with higher clearance
                mean_fc_y = np.mean([p[1] for p in far_center])
                mid_y_far = -1.5 if mean_fc_y >= 0.0 else +1.5
                mode_far = "U_TURN_FORCE"
        elif has_far_left and has_far_right:
            y_fl = np.mean(sorted(far_left)[:max(1, len(far_left) // 4)])
            y_fr = np.mean(sorted(far_right, reverse=True)[:max(1, len(far_right) // 4)])
            mid_y_far = (y_fl + y_fr) / 2.0
            mode_far = "BOTH"
        elif has_far_left:
            y_fl = np.mean(sorted(far_left)[:max(1, len(far_left) // 4)])
            mid_y_far = y_fl - 1.15
            mode_far = "LEFT_ONLY"
        elif has_far_right:
            y_fr = np.mean(sorted(far_right, reverse=True)[:max(1, len(far_right) // 4)])
            mid_y_far = y_fr + 1.15
            mode_far = "RIGHT_ONLY"
        else:
            mid_y_far = mid_y_near
            mode_far = "NONE"

        # Combine Near & Far Slice Steering Offset
        target_y = 0.35 * mid_y_near + 0.65 * mid_y_far
        angular_z = kp_center * target_y

        # Keep cruising speed CONSTANT at target_speed (0.5 m/s) - Never slow down or stop!
        linear_x = target_speed

        # 4. Incline Speed Compensation
        speed_factor = 1.0
        if self.has_imu and linear_x > 0.0:
            speed_factor = 1.0 - (self.current_pitch * pitch_scale_factor)
            speed_factor = np.clip(speed_factor, min_speed_factor, max_speed_factor)
            linear_x = linear_x * speed_factor

        # Clamp angular velocity limit
        angular_z = np.clip(angular_z, -max_angular_speed, max_angular_speed)

        # Publish velocities
        cmd_msg = Twist()
        cmd_msg.linear.x = float(linear_x)
        cmd_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(cmd_msg)

        # Log status
        self.get_logger().info(
            f"[U-CORRIDOR] Near:{mode_near} Far:{mode_far} | "
            f"mid_near={mid_y_near:.2f}m mid_far={mid_y_far:.2f}m target_y={target_y:.2f}m | "
            f"Cmd: v={linear_x:.2f} m/s, w={angular_z:.2f} rad/s",
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
