"""
SRA System Launch File

Starts all current SRA nodes in the correct order:
  1. sra_state_machine_node — system state authority
  2. sra_tts_node           — offline Piper speech output
  3. sra_command_parser     — LLM interface
  4. sra_task_manager       — inventory, queue, task creation
  5. sra_delivery_sim       — simulated robot executor
  6. sra_voice_input        — microphone capture and transcription

Usage:
  source ~/sra_project/.env
  ros2 launch sra_bringup sra_system.launch.py

Optional arguments (override at launch time):
  ros2 launch sra_bringup sra_system.launch.py voice:=false
  ros2 launch sra_bringup sra_system.launch.py sim_time:=10.0
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ----
    # Launch arguments — can be overridden from the command line
    # ----
    voice_arg = DeclareLaunchArgument(
        'voice',
        default_value='true',
        description='Set to false to disable voice input (text-only mode)'
    )

    sim_time_arg = DeclareLaunchArgument(
        'sim_time',
        default_value=os.getenv('SRA_DELIVERY_SIM_TIME', '5.0'),
        description='Simulated delivery travel time in seconds'
    )

    # ----
    # Node definitions
    # Each node maps to one ros2 run command.
    # output='screen' shows the node logs in this terminal.
    # ----

    state_machine_node = Node(
        package='sra_state_machine',
        executable='state_machine',
        name='sra_state_machine',
        output='screen',
        emulate_tty=True,
    )

    command_parser_node = Node(
        package='sra_llm',
        executable='command_parser',
        name='sra_command_parser',
        output='screen',
        emulate_tty=True,   # keeps coloured log output
    )

    tts_node = Node(
        package='sra_tts',
        executable='tts_node',
        name='sra_tts_node',
        output='screen',
        emulate_tty=True,
    )
    
    task_manager_node = Node(
        package='sra_task_manager',
        executable='task_manager',
        name='sra_task_manager',
        output='screen',
        emulate_tty=True,
    )

    delivery_sim_node = Node(
        package='sra_delivery_sim',
        executable='delivery_sim',
        name='sra_delivery_sim',
        output='screen',
        emulate_tty=True,
    )

    # Voice input starts 3 seconds after the others so the LLM model
    # is fully loaded before the first voice command can arrive.
    voice_input_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='sra_voice_input',
                executable='voice_input',
                name='sra_voice_input',
                output='screen',
                emulate_tty=True,
                condition=IfCondition(LaunchConfiguration('voice')),
            )
        ]
    )

    return LaunchDescription([
        # Arguments
        voice_arg,
        sim_time_arg,

        # Startup message
        LogInfo(msg='===='),
        LogInfo(msg='  SRA System starting...'),
        LogInfo(msg='===='),

        # Nodes
        state_machine_node,
        tts_node,
        command_parser_node,
        task_manager_node,
        delivery_sim_node,
        voice_input_node,
    ])
