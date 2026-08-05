#!/usr/bin/env python3
"""
SRA Simulated Delivery Executor Node (v2)

Changes from v1:
- Credentials read from environment variables
- Atomic DB finalization (autocommit=False, explicit commit/rollback)
- Publishes task status updates to /sra/tasks/status for task manager queue
- Simulated delivery time configurable via SRA_DELIVERY_SIM_TIME env var

Bug fixes vs submitted draft:
- finalize_task_in_db used DELETE ... LIMIT 1 which PostgreSQL does not
  support. Fixed to use a subquery: DELETE WHERE id = (SELECT id ... LIMIT 1).
- complete_delivery redundantly re-published 'in_progress' status at the
  start of the completion phase. Removed — in_progress is published in
  task_callback when delivery actually starts.
"""

import json
import os
from datetime import datetime, timezone

import psycopg2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host":     os.getenv("SRA_DB_HOST",     "localhost"),
    "dbname":   os.getenv("SRA_DB_NAME",     "sra_db"),
    "user":     os.getenv("SRA_DB_USER",     "sra_user"),
    "password": os.getenv("SRA_DB_PASSWORD", "123456"),
    "port":     int(os.getenv("SRA_DB_PORT", "5432")),
}

SIMULATED_DELIVERY_SECONDS = float(os.getenv("SRA_DELIVERY_SIM_TIME", "5.0"))


class DeliverySimNode(Node):

    def __init__(self):
        super().__init__("sra_delivery_sim")

        self.conn = self.connect_db()

        # Track the single active delivery and its one-shot timer
        self.active_task    = None
        self.delivery_timer = None

        self.heartbeat_timer = None

        # --- Subscribers ---
        self.sub = self.create_subscription(String, "/sra/tasks/active", self.task_callback, 10)
        self.sub = self.create_subscription(String, "/sra/system/emergency_stop", self.emergency_stop_callback, 10)
        self.sub = self.create_subscription(String, "/sra/operator/recover", self.on_recover,10)

        # --- Publishers ---
        self.completed_pub = self.create_publisher(String, "/sra/tasks/completed", 10)
        self.status_pub    = self.create_publisher(String, "/sra/tasks/status",    10)
        self.alert_pub     = self.create_publisher(String, "/sra/alerts/events",   10)

        self.get_logger().info(
            f"Delivery simulator ready. "
            f"Simulated travel time: {SIMULATED_DELIVERY_SECONDS}s"
        )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def connect_db(self):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = False
            self.get_logger().info('Database connected.')
            return conn
        except Exception as exc:
            self.get_logger().error(f'Database connection failed: {exc}')
            raise RuntimeError(f'Cannot start without database: {exc}')

    # ------------------------------------------------------------------
    # Task handling
    # ------------------------------------------------------------------

    def task_callback(self, msg: String):
        """Receive a task from the task manager and begin simulated delivery."""
        try:
            task = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Malformed task JSON on /sra/tasks/active")
            return

        task_id = task["task_id"]

        # Reject if already busy — task manager should not send a second task
        # while one is active, but guard defensively
        if self.active_task is not None:
            self.get_logger().warn(
                f"Robot busy with task {self.active_task['task_id']}. "
                f"Rejecting task {task_id}."
            )
            self._publish_status(task_id, "rejected")
            return

        self.active_task = task

        # Notify task manager and update database
        self._publish_status(task_id, "accepted")
        self.update_task_status(task_id, "in_progress")

        self.get_logger().info(
            f"[Task {task_id}] Starting simulated delivery: "
            f"{task['quantity']}x {task['config_code']} → {task['destination']}"
        )
        self.publish_alert(
            "info",
            f"[Tarea {task_id}] Robot en camino → {task['destination']}"
        )

        # Start heartbeats every 5 seconds (must be less than the timeout)
        self.heartbeat_timer = self.create_timer(5.0, self._publish_heartbeat)

        # Start the one-shot delivery timer
        self.delivery_timer = self.create_timer(
            SIMULATED_DELIVERY_SECONDS,
            self.complete_delivery
        )

    def complete_delivery(self):
        """
        Called by the timer when simulated travel time has elapsed.
        Finalizes the task atomically in the database and publishes the result.
        """
        # Cancel first to prevent the periodic timer from firing again
        if self.delivery_timer is not None:
            self.delivery_timer.cancel()
            self.delivery_timer = None

        if self.heartbeat_timer is not None:
            self.heartbeat_timer.cancel()
            self.heartbeat_timer = None

        if self.active_task is None:
            self.get_logger().warn("complete_delivery fired but no active task — ignoring.")
            return
        
        task    = self.active_task
        task_id = task["task_id"]

        self.get_logger().info(
            f"[Task {task_id}] Delivery complete. Updating database..."
        )

        success = self.finalize_task_in_db(task)

        if success:
            completed = {
                "task_id":     task_id,
                "config_code": task["config_code"],
                "quantity":    task["quantity"],
                "destination": task["destination"],
                "status":      "completed",
                "timestamp":   datetime.now(timezone.utc).isoformat(),
            }
            msg = String()
            msg.data = json.dumps(completed)
            self.completed_pub.publish(msg)

            self._publish_status(task_id, "completed")
            self.get_logger().info(f"[Task {task_id}] ✓ Task completed and published.")
            self.publish_alert(
                "info",
                f"[Tarea {task_id}] Entregado: "
                f"{task['quantity']}x {task['config_code']} → {task['destination']}"
            )
        else:
            self._publish_status(task_id, "failed")
            self.get_logger().error(
                f"[Task {task_id}] Database update failed after delivery."
            )
            self.publish_alert(
                "error",
                f"[Tarea {task_id}] Entrega completada pero fallo al actualizar BD"
            )

        # Free the executor for the next task
        self.active_task = None

    def emergency_stop_callback(self, msg: String) -> None:
        """
        Immediately cancel the active delivery timer.
        The task is marked cancelled so task_manager can release its reservation.
        finalize_task_in_db is intentionally skipped — the physical delivery
        did not complete, so inventory must not be deducted.
        """
        if self.active_task is None:
            self.get_logger().info("Emergency stop received — no active task.")
            return

        task_id = self.active_task["task_id"]
        self.get_logger().warn(
            f"EMERGENCY STOP — cancelling task {task_id} mid-delivery."
        )

        # Kill the timer before it fires complete_delivery()
        if self.delivery_timer is not None:
            self.delivery_timer.cancel()
            self.delivery_timer = None

        if self.heartbeat_timer is not None:
            self.heartbeat_timer.cancel()
            self.heartbeat_timer = None

        # Notify task_manager so it can clean up the queue
        self._publish_status(task_id, "cancelled")

        self.publish_alert(
            "error",
            f"[Tarea {task_id}] Entrega cancelada por parada de emergencia."
        )
        self.active_task = None

    def on_recover(self, msg: String):
        """Simulate physical homing on recovery."""
        self.get_logger().info("Recovery received: Homing simulator/hardware...")
        # (Future: send UR10e to home joint positions)
        self.active_task = None
        
    # ------------------------------------------------------------------
    # Database operations
    # ------------------------------------------------------------------

    def finalize_task_in_db(self, task: dict) -> bool:
        """
        Four operations wrapped in a single atomic transaction:
          1. Mark task as completed with timestamp
          2. Deduct delivered quantity from inventory
          3. Remove the inventory reservation for this task
          4. Log the delivery as a transaction record

        If any step fails the entire transaction is rolled back, leaving
        the database in a consistent state.

        FIX: PostgreSQL does not support DELETE ... LIMIT 1 directly.
        The reservation deletion uses a subquery to identify the target row
        by primary key before deleting it.
        """
        try:
            with self.conn.cursor() as cur:

                # 1. Mark task completed
                cur.execute("""
                    UPDATE tasks
                    SET status       = 'completed',
                        completed_at = NOW()
                    WHERE id = %s
                """, (task["task_id"],))

                # 2. Deduct inventory
                cur.execute("""
                    UPDATE inventory
                    SET quantity     = quantity - %s,
                        last_updated = NOW()
                    FROM configurations
                    WHERE inventory.config_id    = configurations.id
                      AND configurations.config_code = %s
                """, (task["quantity"], task["config_code"]))

                # 3. Remove reservation scoped to this task_id (Bug #3 fix)
                cur.execute("""
                    DELETE FROM inventory_reservations
                    WHERE task_id = %s
                """, (task["task_id"],))

                # 4. Log transaction
                cur.execute("""
                    INSERT INTO transactions
                        (event_type, config_code, quantity, task_id, station, notes)
                    VALUES
                        ('delivered', %s, %s, %s, %s, 'Simulated delivery')
                """, (
                    task["config_code"],
                    task["quantity"],
                    task["task_id"],
                    task["destination"],
                ))

            self.conn.commit()
            return True

        except Exception as exc:
            self.conn.rollback()
            self.get_logger().error(f"Database finalization failed: {exc}")
            return False

    def update_task_status(self, task_id: int, status: str) -> None:
        """Update task status and set started_at when transitioning to in_progress."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE tasks
                    SET status     = %s,
                        started_at = CASE
                                       WHEN %s = 'in_progress' THEN NOW()
                                       ELSE started_at
                                     END
                    WHERE id = %s
                """, (status, status, task_id))
            self.conn.commit()
        except Exception as exc:
            self.conn.rollback()
            self.get_logger().error(f"Status update failed: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_status(self, task_id: int, status: str) -> None:
        """Publish a task lifecycle update for the task manager's queue logic."""
        status_msg = {
            "task_id":   task_id,
            "status":    status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        msg = String()
        msg.data = json.dumps(status_msg)
        self.status_pub.publish(msg)

    def publish_alert(self, level: str, message: str) -> None:
        alert = {
            "level":     level,
            "message":   message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        msg = String()
        msg.data = json.dumps(alert)
        self.alert_pub.publish(msg)

    def destroy_node(self):
        if self.conn:
            self.conn.close()
        super().destroy_node()

    def _publish_heartbeat(self):
        if self.active_task:
            self._publish_status(self.active_task["task_id"], "in_progress")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = DeliverySimNode()
    except RuntimeError as exc:
        print(f'[FATAL] Node failed to start: {exc}')
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
