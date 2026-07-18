# Teknofest Rover Package

This is the main package for the autonomous rover navigation, planning, and control stack developed for the Teknofest competition.

## Prerequisites

> [!IMPORTANT]
> **Important Dependency:** You must have the **`rover_sim`** package installed and configured in your ROS 2 workspace. It is a critical prerequisite for launching the Gazebo simulation environment.

Ensure your workspace structure contains both packages under `src/`:
```bash
rover_ws/
└── src/
    ├── rover_sim/    # Simulation, worlds, models, and spawning configurations
    └── teknofest/    # This package (navigation, mapping, localization, and control)
```

## Setup and Building

1. Make sure all ROS 2 dependencies are installed and the workspace is fully built:
   ```bash
   cd ~/rover_ws
   colcon build --symlink-install
   source install/setup.bash
   ```

## Launching the Simulation

To launch the Gazebo simulation using the included launcher in this package:

```bash
ros2 launch teknofest gazebo.launch.py
```

This launch file includes and executes the underlying Gazebo simulator defined in the `rover_sim` package, with automatic argument forwarding.
