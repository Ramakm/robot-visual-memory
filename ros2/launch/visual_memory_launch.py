"""ROS2 launch file for the visual memory node."""

try:
    from launch import LaunchDescription
    from launch_ros.actions import Node

    def generate_launch_description() -> LaunchDescription:
        return LaunchDescription(
            [
                Node(
                    package="robot_visual_memory",
                    executable="visual_memory_node",
                    name="visual_memory_node",
                    parameters=[
                        {"config_path": "config/default.yaml"},
                        {"room_id": "default"},
                    ],
                    output="screen",
                ),
            ]
        )

except ImportError:
    pass
