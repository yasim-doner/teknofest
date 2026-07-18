#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32

class CmdSwitchNode(Node):
    """
    ROS 2 Node that switches/multiplexes cmd_vel topics depending on the active stage.
    By default, it forwards fallow_corridor's cmd_vel.
    When a specific stage is active (e.g. Stage 6 for dynamic_obstacle), it forwards
    that stage's cmd_vel.
    When the active stage node publishes a release message, it reverts back to fallow_corridor.
    """
    def __init__(self):
        super().__init__('cmd_switch')

        # Current active stage ID (0 corresponds to fallow_corridor / default)
        self.active_stage = 0

        # Mapping of stage ID to the source cmd_vel topic
        self.stage_topics = {
            6: '/dynamic_obstacle/cmd_vel',
        }

        # Subscribers for candidate command velocities
        self.fallow_sub = self.create_subscription(
            Twist,
            '/fallow_corridor/cmd_vel',
            self.fallow_callback,
            10
        )
        
        self.dynamic_sub = self.create_subscription(
            Twist,
            '/dynamic_obstacle/cmd_vel',
            self.dynamic_callback,
            10
        )

        # Subscriber for stage detection
        self.stage_sub = self.create_subscription(
            Int32,
            '/teknofest/stage_id',
            self.stage_callback,
            10
        )

        # Subscriber for release signal from active nodes
        self.release_sub = self.create_subscription(
            Int32,
            '/teknofest/release',
            self.release_callback,
            10
        )

        # Publisher for the active control cmd_vel
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/rover/cmd_vel',
            10
        )

        self.get_logger().info("=== Cmd Switch Node Initialized ===")
        self.get_logger().info("Default active source: /fallow_corridor/cmd_vel (Stage 0)")

    def fallow_callback(self, msg: Twist):
        # Route fallow_corridor if no specific stage node is active
        if self.active_stage not in self.stage_topics:
            self.cmd_vel_pub.publish(msg)

    def dynamic_callback(self, msg: Twist):
        # Route dynamic_obstacle only if Stage 6 is active
        if self.active_stage == 6:
            self.cmd_vel_pub.publish(msg)

    def stage_callback(self, msg: Int32):
        stage_id = msg.data
        if stage_id != 0 and stage_id != self.active_stage:
            self.active_stage = stage_id
            target_topic = self.stage_topics.get(stage_id, '/fallow_corridor/cmd_vel')
            self.get_logger().info(f"Stage {stage_id} detected. Switching active control to: {target_topic}")

    def release_callback(self, msg: Int32):
        released_stage = msg.data
        if released_stage == self.active_stage:
            self.get_logger().info(f"Stage {released_stage} node released control. Reverting to fallow_corridor.")
            self.active_stage = 0

def main(args=None):
    rclpy.init(args=args)
    node = CmdSwitchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
