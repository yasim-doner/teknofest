import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

# 11 Sign Poses for Spawning the Robot
# Format: (initial_x, initial_y, initial_z, initial_yaw)
SIGN_POSES = {
    1:  (16.0,  0.0, 0.5, 3.1416),  # Water Crossing & Rain (Su Geçişi ve Yağmur)
    2:  (11.0,  0.0, 0.5, 3.1416),  # Stony / Gravel Road (Taşlı / Çakıllı Yol)
    3:  (6.0,   0.0, 0.5, 3.1416),  # Side Slope (Yan Eğim)
    4:  (2.0,  10.0, 0.5, 0.0000),  # Vertical Obstacle (Dik Engel)
    5:  (7.0,  10.0, 0.5, 0.0000),  # Traffic Cones (Trafik Konileri)
    6:  (14.0, 10.0, 0.5, 0.0000),  # Sliding Barrier (Kayar Engel)
    7:  (18.0, 20.0, 0.5, 3.1416),  # Rough Terrain (Engebeli Arazi)
    8:  (10.0, 20.0, 1.5, 3.1416),  # Upward / Downward Slope (Dik Eğimler)
    9:  (7.8,  20.0, 1.5, 3.1416),  # Targeting & Shooting (Atış)
    10: (4.8,  20.0, 1.5, 3.1416),  # Final Stage / Exit
    11: (20.0, -8.0, 0.5, 3.1416),  # Acceleration Track (Hızlanma Parkuru)
}

def launch_setup(context, *args, **kwargs):
    # Retrieve configurations
    stage_val = context.launch_configurations.get('stage', '0')
    
    # Defaults
    x_val = context.launch_configurations.get('initial_x', '17.5')
    y_val = context.launch_configurations.get('initial_y', '0.0')
    z_val = context.launch_configurations.get('initial_z', '0.5')
    yaw_val = context.launch_configurations.get('initial_yaw', '3.14159')

    # If stage is selected (1 to 11), override defaults with the sign start pose
    try:
        stage = int(stage_val)
        if stage in SIGN_POSES:
            px, py, pz, pyaw = SIGN_POSES[stage]
            x_val = str(px)
            y_val = str(py)
            z_val = str(pz)
            yaw_val = str(pyaw)
            print(f"\n[sim.launch.py] Spawning at Stage {stage} Start: X={x_val}, Y={y_val}, Yaw={yaw_val}\n")
    except ValueError:
        pass

    rover_sim_launch_path = PathJoinSubstitution([
        FindPackageShare('rover_sim'),
        'launch',
        'gazebo.launch.py'
    ])

    include_rover_sim_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rover_sim_launch_path),
        launch_arguments={
            'initial_x': x_val,
            'initial_y': y_val,
            'initial_z': z_val,
            'initial_yaw': yaw_val
        }.items()
    )

    # Corridor Follower Node
    fallow_corridor_node = Node(
        package='teknofest',
        executable='fallow_corridor.py',
        name='fallow_corridor',
        output='screen'
    )

    # Dynamic Obstacle Node
    dynamic_obstacle_node = Node(
        package='teknofest',
        executable='dynamic_obstacle.py',
        name='dynamic_obstacle',
        output='screen'
    )

    # Sign Detector Node
    sign_detect_node = Node(
        package='teknofest',
        executable='sign_detect.py',
        name='sign_detect',
        output='screen'
    )

    return [
        include_rover_sim_gazebo,
        fallow_corridor_node,
        #dynamic_obstacle_node,
        sign_detect_node
    ]

def generate_launch_description():
    # Declare launch arguments
    stage_arg = DeclareLaunchArgument(
        name="stage",
        default_value="0",
        description="Select Stage (1-11) to spawn robot at the start of that sign. 0 for default."
    )

    x_arg = DeclareLaunchArgument(
        name="initial_x",
        default_value="17.5",
        description="X coordinate where rover would be spawned"
    )

    y_arg = DeclareLaunchArgument(
        name="initial_y",
        default_value="0.0",
        description="Y coordinate where rover would be spawned"
    )

    z_arg = DeclareLaunchArgument(
        name="initial_z",
        default_value="0.5",
        description="Height rover would be spawned"
    )

    yaw_arg = DeclareLaunchArgument(
        name="initial_yaw",
        default_value="3.14159",
        description="Yaw rotation where rover would be spawned"
    )

    return LaunchDescription([
        stage_arg,
        x_arg,
        y_arg,
        z_arg,
        yaw_arg,
        OpaqueFunction(function=launch_setup)
    ])
