import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Find the rover_sim package and locate gazebo.launch.py
    rover_sim_launch_path = PathJoinSubstitution([
        FindPackageShare('rover_sim'),
        'launch',
        'gazebo.launch.py'
    ])

    # Include rover_sim's gazebo launch description
    # This will automatically forward any passed arguments (like initial position, gui, world, etc.)
    include_rover_sim_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rover_sim_launch_path)
    )

    return LaunchDescription([
        include_rover_sim_gazebo
    ])
