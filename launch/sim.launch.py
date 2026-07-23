import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
    8:  (11.5, 20.0, 0.5, 3.1416),  # Rampanın önündeki alt düzlük
    9:  (7.8,  20.0, 1.5, 3.1416),  # Targeting & Shooting (Atış)
    10: (4.8,  20.0, 1.5, 3.1416),  # Final Stage / Exit
    11: (20.0, -8.0, 0.5, 3.1416),  # Acceleration Track (Hızlanma Parkuru)
}


def launch_setup(context, *args, **kwargs):
    # Retrieve configurations
    stage_val = context.launch_configurations.get("stage", "0")

    # Stage değeri dönüştürülemezse kullanılacak varsayılan değer
    stage = 0

    # Defaults
    x_val = context.launch_configurations.get("initial_x", "17.5")
    y_val = context.launch_configurations.get("initial_y", "0.0")
    z_val = context.launch_configurations.get("initial_z", "0.5")
    yaw_val = context.launch_configurations.get("initial_yaw", "3.14159")

    # If stage is selected (1 to 11), override defaults with the sign start pose
    try:
        stage = int(stage_val)

        if stage in SIGN_POSES:
            px, py, pz, pyaw = SIGN_POSES[stage]

            x_val = str(px)
            y_val = str(py)
            z_val = str(pz)
            yaw_val = str(pyaw)

            print(
                f"\n[sim.launch.py] "
                f"Spawning at Stage {stage} Start: "
                f"X={x_val}, Y={y_val}, Yaw={yaw_val}\n"
            )

    except (ValueError, TypeError):
        stage = 0

    rover_sim_launch_path = PathJoinSubstitution([
        FindPackageShare("rover_sim"),
        "launch",
        "gazebo.launch.py",
    ])

    include_rover_sim_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rover_sim_launch_path),
        launch_arguments={
            "initial_x": x_val,
            "initial_y": y_val,
            "initial_z": z_val,
            "initial_yaw": yaw_val,
        }.items(),
    )

    params_file_path = os.path.join(
        get_package_share_directory("teknofest"),
        "params",
        "teknofest_params.yaml",
    )

    # Corridor Follower Node
    fallow_corridor_node = Node(
        package="teknofest",
        executable="fallow_corridor.py",
        name="fallow_corridor",
        output="screen",
        parameters=[
            params_file_path,
            {
                "use_sim_time": True,
                "initial_stage": stage,
            },
        ],
        arguments=[
            "--ros-args",
            "--log-level",
            "warn",
        ],
    )

    # Cone Avoid Node
    cone_avoid_node = Node(
        package="teknofest",
        executable="cone_avoid.py",
        name="cone_avoid",
        output="screen",
        parameters=[
            params_file_path,
            {
                "use_sim_time": True,
                "initial_stage": stage,
            },
        ],
    )

    # Dynamic Obstacle Node
    dynamic_obstacle_node = Node(
        package="teknofest",
        executable="dynamic_obstacle.py",
        name="dynamic_obstacle",
        output="screen",
        parameters=[
            params_file_path,
            {
                "use_sim_time": True,
            },
        ],
    )

    # Sign Detector Node
    sign_detect_node = Node(
        package="teknofest",
        executable="sign_detect.py",
        name="sign_detect",
        output="screen",
        parameters=[
            params_file_path,
            {
                "use_sim_time": True,
                "initial_stage": stage,
                "image_topic": "/rover/camera/image_raw",

                # Hafif connected-components algılama ayarları.
                "min_component_area": 30,
                "max_component_area_ratio": 0.15,
                "min_candidate_size": 8,
                "max_candidate_size": 220,
                "min_aspect": 0.45,
                "max_aspect": 1.75,
                "min_white_ratio": 0.08,
                "min_red_fill_ratio": 0.02,

                # Fiziksel levha yaklaşma / uzaklaşma eşikleri.
                "enter_radius": 14,
                "exit_radius": 9,
                "enter_frames": 2,
                "exit_frames": 3,

                # Alt bölümdeki konileri stage levhası sayma.
                "max_stage_candidate_y_ratio": 0.78,
                "upper_enter_radius": 10,

                # Stage 5 -> 6 ve Stage 6 -> 7 için simülasyon yedeği.
                "use_odom_fallback": True,
                "stage5_to6_x": 12.5,
                "stage5_lane_y": 10.0,
                "stage5_lane_tolerance": 3.0,
                "stage6_to7_min_x": 15.0,
                "stage6_to7_y": 17.0,

                # Stage 8 sonrasında sonraki yakın levha STOP kabul edilir.
                "stop_guard_seconds": 3.0,
                "diagnostic_every_frames": 5,

                # Art arda stage'lerin aynı nesneden üretilmesini engeller.
                "min_stage_travel_distance": 1.5,
                "min_stage_interval_seconds": 2.0,

                # Sıradaki stage'in gerçek parkur konumuna yaklaşmadan
                # stage artırılmaz.
                "use_stage_geofence": True,
                "stage_geofence_radius": 3.0,

                # Normal Teknofest parkuru Stage 10'da biter.
                # Stage 11 ayrı hızlanma parkurudur.
                "final_stage": 10,
                "stop_after_final_stage": True,
            },
        ],
    )

    # Stage 8 Ramp + Laser Task Node
    laser_target_node = Node(
        package="teknofest",
        executable="laser_target.py",
        name="laser_target",
        output="screen",
        parameters=[
            params_file_path,
            {
                "use_sim_time": True,
                "active_stage": 8,
                "initial_stage": stage,
                "approach_speed": 0.35,
                "ramp_speed": 0.45,
                "ramp_boost_speed": 0.55,
                "ramp_boost_delay": 2.0,

                # Rampa algılandıktan sonra rover rampanın üzerinde
                # tamamen hareketsiz bekler, sonra çıkmaya devam eder.
                "ramp_pause_duration": 2.0,

                # Rampa algılandıktan sonra yaklaşık 20 cm daha ilerler,
                # ardından rampanın üzerinde 2 saniye bekler.
                "ramp_creep_duration": 3.0,
                "ramp_creep_speed": 0.25,

                "search_speed": 0.08,
                "ramp_pitch_threshold_deg": 8.0,
                "flat_pitch_threshold_deg": 3.0,
                "ramp_confirm_frames": 5,
                "flat_confirm_frames": 12,
                "stop_sign_wait_timeout": 2.0,
                "stop_duration": 2.2,
                "laser_duration": 1.2,
            },
        ],
    )

    # Command Switch Node
    cmd_switch_node = Node(
        package="teknofest",
        executable="cmd_switch.py",
        name="cmd_switch",
        output="screen",
        parameters=[
            params_file_path,
            {
                "use_sim_time": True,
                "initial_stage": stage,
            },
        ],
    )

    # RViz2 Node
    rviz_config_path = os.path.join(
        get_package_share_directory("teknofest"),
        "params",
        "tekno.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            rviz_config_path,
        ],
        parameters=[
            {
                "use_sim_time": True,
            },
        ],
        output="screen",
    )

    return [
        include_rover_sim_gazebo,
        fallow_corridor_node,
        cone_avoid_node,
        dynamic_obstacle_node,
        sign_detect_node,
        laser_target_node,
        cmd_switch_node,
        rviz_node,
    ]


def generate_launch_description():
    # Declare launch arguments
    stage_arg = DeclareLaunchArgument(
        name="stage",
        default_value="0",
        description=(
            "Select Stage (1-11) to spawn robot at the start "
            "of that sign. 0 for default."
        ),
    )

    x_arg = DeclareLaunchArgument(
        name="initial_x",
        default_value="17.5",
        description="X coordinate where rover would be spawned",
    )

    y_arg = DeclareLaunchArgument(
        name="initial_y",
        default_value="0.0",
        description="Y coordinate where rover would be spawned",
    )

    z_arg = DeclareLaunchArgument(
        name="initial_z",
        default_value="0.5",
        description="Height rover would be spawned",
    )

    yaw_arg = DeclareLaunchArgument(
        name="initial_yaw",
        default_value="3.14159",
        description="Yaw rotation where rover would be spawned",
    )

    return LaunchDescription([
        stage_arg,
        x_arg,
        y_arg,
        z_arg,
        yaw_arg,
        OpaqueFunction(function=launch_setup),
    ])
