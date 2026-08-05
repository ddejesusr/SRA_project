#!/usr/bin/env python3
"""
SRA Simulated Delivery Executor Node

Pretends to be the mobile robot + UR10e during Phase 1 (office testing).
Receives tasks, waits to simulate travel time, then updates the database
and publishes task completion.

In Phase 2, this node gets replaced by the real robot executor.
"""

import json
import psycopg2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from datetime import datetime, timezone


DB_CONFIG = {
    "host":     "localhost",
    "dbname":   "sra_db",
    "user":     "sra_user",
    "password": "123456",
    "port":     5432
}

# How many seconds to simulate delivery travel time
SIMULATED_DELIVERY_SECONDS = 5.0


class DeliverySimNode(Node):

    def __init__(self):
        super().__init__('sra_delivery_sim')

        # Database connection
        self.conn = self.connect_db()

        # Track the active task and its timer
        # Only one delivery happens at a time (robot has one platform)
        self.active_task = None
        self.delivery_timer = None

        # --- Subscribers ---
        # Listens for new tasks from the task manager
        self.sub = self.create_subscription(
            String,
            '/sra/tasks/active',
            self.task_callback,
            10
        )

        # --- Publishers ---
        # Announces completed deliveries
        self.completed_pub = self.create_publisher(
            String,
            '/sra/tasks/completed',
            10
        )

        # Sends alerts to the dashboard and other nodes
        self.alert_pub = self.create_publisher(
            String,
            '/sra/alerts/events',
            10
        )

        self.get_logger().info(
            f'Delivery simulator ready. '
            f'Simulated travel time: {SIMULATED_DELIVERY_SECONDS}s'
        )

    def connect_db(self):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = True
            self.get_logger().info('Database connected.')
            return conn
        except Exception as e:
            self.get_logger().error(f'Database connection failed: {e}')
            return None

    def task_callback(self, msg: String):
        """Called every time a new task arrives on /sra/tasks/active."""

        # Parse the incoming JSON
        try:
            task = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Received malformed task JSON.')
            return

        # If robot is already busy, reject the new task
        # (In the real system, tasks would queue — we keep it simple for now)
        if self.active_task is not None:
            self.get_logger().warn(
                f'Robot busy with task {self.active_task["task_id"]}. '
                f'Task {task["task_id"]} ignored.'
            )
            return

        self.active_task = task
        task_id     = task['task_id']
        config_code = task['config_code']
        quantity    = task['quantity']
        destination = task['destination']

        self.get_logger().info(
            f'[Task {task_id}] Starting simulated delivery: '
            f'{quantity}x {config_code} → {destination}'
        )

        # Mark task as in_progress in the database
        self.update_task_status(task_id, 'in_progress')

        # Publish an alert so the dashboard knows delivery started
        self.publish_alert(
            'info',
            f'[Task {task_id}] Robot departing → {destination}'
        )

        # Start the delivery timer
        # After SIMULATED_DELIVERY_SECONDS, call self.complete_delivery
        # timer_period, callback, autoreset=False means it fires once only
        self.delivery_timer = self.create_timer(
            SIMULATED_DELIVERY_SECONDS,
            self.complete_delivery
        )

    def complete_delivery(self):
        """Called automatically by the timer when simulated delivery finishes."""

        # Cancel the timer so it doesn't fire again
        self.delivery_timer.cancel()
        self.delivery_timer = None

        task        = self.active_task
        task_id     = task['task_id']
        config_code = task['config_code']
        quantity    = task['quantity']
        destination = task['destination']

        self.get_logger().info(
            f'[Task {task_id}] Delivery complete. Updating database...'
        )

        # Update database: mark task completed, deduct inventory, log transaction
        success = self.finalize_task_in_db(task_id, config_code, quantity, destination)

        if success:
            # Publish completion message for other nodes to react to
            completed = {
                "task_id":     task_id,
                "config_code": config_code,
                "quantity":    quantity,
                "destination": destination,
                "status":      "completed",
                "timestamp":   datetime.now(timezone.utc).isoformat()
            }
            msg = String()
            msg.data = json.dumps(completed)
            self.completed_pub.publish(msg)

            self.get_logger().info(
                f'[Task {task_id}] ✓ Task completed and published.'
            )
            self.publish_alert(
                'info',
                f'[Task {task_id}] Delivered {quantity}x {config_code} to {destination}'
            )
        else:
            self.get_logger().error(
                f'[Task {task_id}] Database update failed after delivery.'
            )
            self.publish_alert(
                'error',
                f'[Task {task_id}] Delivery done but database update failed!'
            )

        # Clear active task — robot is free for next task
        self.active_task = None

    def finalize_task_in_db(self, task_id, config_code, quantity, destination):
        """
        Three database operations that must all succeed together:
        1. Mark task as completed
        2. Deduct delivered quantity from inventory
        3. Log the delivery as a transaction
        """
        try:
            with self.conn.cursor() as cur:

                # 1. Update task status and record completion time
                cur.execute("""
                    UPDATE tasks
                    SET status       = 'completed',
                        completed_at = NOW()
                    WHERE id = %s
                """, (task_id,))

                # 2. Deduct inventory
                cur.execute("""
                    UPDATE inventory
                    SET quantity     = quantity - %s,
                        last_updated = NOW()
                    FROM configurations
                    WHERE inventory.config_id  = configurations.id
                      AND configurations.config_code = %s
                """, (quantity, config_code))

                # 3. Log the transaction
                cur.execute("""
                    INSERT INTO transactions
                        (event_type, config_code, quantity, task_id, station, notes)
                    VALUES
                        ('delivered', %s, %s, %s, %s, 'Simulated delivery')
                """, (config_code, quantity, task_id, destination))

            return True

        except Exception as e:
            self.get_logger().error(f'Database finalization failed: {e}')
            return False

    def update_task_status(self, task_id, status):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE tasks SET status = %s,
                    started_at = CASE WHEN %s = 'in_progress'
                                 THEN NOW() ELSE started_at END
                    WHERE id = %s
                """, (status, status, task_id))
        except Exception as e:
            self.get_logger().error(f'Status update failed: {e}')

    def publish_alert(self, level: str, message: str):
        alert = {
            "level":     level,
            "message":   message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        msg = String()
        msg.data = json.dumps(alert)
        self.alert_pub.publish(msg)

    def destroy_node(self):
        if self.conn:
            self.conn.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DeliverySimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
