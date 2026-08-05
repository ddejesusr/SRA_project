#!/usr/bin/env python3
"""
SRA Task Manager Node
Receives parsed commands, validates against inventory,
creates task records in PostgreSQL, and publishes active tasks.
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone


DB_CONFIG = {
    "host":     "localhost",
    "dbname":   "sra_db",
    "user":     "sra_user",
    "password": "123456",
    "port":     5432
}


class TaskManagerNode(Node):

    def __init__(self):
        super().__init__('sra_task_manager')

        self.conn = self.connect_db()

        # Subscriber: validated commands from LLM parser
        self.sub = self.create_subscription(
            String,
            '/sra/ai/command',
            self.command_callback,
            10
        )

        # Publisher: active task for the robot to execute
        self.task_pub = self.create_publisher(
            String,
            '/sra/tasks/active',
            10
        )

        # Publisher: system alerts and events
        self.alert_pub = self.create_publisher(
            String,
            '/sra/alerts/events',
            10
        )

        self.get_logger().info('Task manager ready.')

    def connect_db(self):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = True
            self.get_logger().info('Database connected.')
            return conn
        except Exception as e:
            self.get_logger().error(f'Database connection failed: {e}')
            return None

    def command_callback(self, msg: String):
        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Received malformed JSON on /sra/ai/command')
            return

        # Ignore commands that failed parsing
        if command.get('error'):
            self.get_logger().warn(
                f'Ignoring failed command: {command["error"]}'
            )
            return

        config_code = command['config_code']
        quantity    = command['quantity']
        destination = command['destination']

        self.get_logger().info(
            f'Processing task: {config_code} x{quantity} → {destination}'
        )

        # Check inventory
        available = self.get_inventory(config_code)
        if available is None:
            self.publish_alert('error', f'Config not found in DB: {config_code}')
            return

        if available < quantity:
            msg_text = (
                f'Insufficient stock for {config_code}: '
                f'requested {quantity}, available {available}'
            )
            self.get_logger().warn(msg_text)
            self.publish_alert('warning', msg_text)
            return

        # Create task in database
        task_id = self.create_task(config_code, quantity, destination)
        if task_id is None:
            self.publish_alert('error', 'Failed to create task in database')
            return

        # Publish task for robot execution
        task = {
            "task_id":     task_id,
            "config_code": config_code,
            "quantity":    quantity,
            "destination": destination,
            "status":      "pending"
        }
        task_msg = String()
        task_msg.data = json.dumps(task)
        self.task_pub.publish(task_msg)

        self.get_logger().info(
            f'Task {task_id} created and published → {destination}'
        )
        self.publish_alert(
            'info',
            f'Task {task_id}: deliver {quantity}x {config_code} to {destination}'
        )

    def get_inventory(self, config_code: str):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT i.quantity
                    FROM inventory i
                    JOIN configurations c ON i.config_id = c.id
                    WHERE c.config_code = %s
                """, (config_code,))
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            self.get_logger().error(f'Inventory query failed: {e}')
            return None

    def create_task(self, config_code, quantity, destination):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tasks (config_code, quantity, destination, status)
                    VALUES (%s, %s, %s, 'pending')
                    RETURNING id
                """, (config_code, quantity, destination))
                return cur.fetchone()[0]
        except Exception as e:
            self.get_logger().error(f'Task creation failed: {e}')
            return None

    def publish_alert(self, level: str, message: str):
        alert = {
            "level":     level,
            "message":   message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        alert_msg = String()
        alert_msg.data = json.dumps(alert)
        self.alert_pub.publish(alert_msg)

    def destroy_node(self):
        if self.conn:
            self.conn.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
