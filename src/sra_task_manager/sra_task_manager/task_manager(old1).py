#!/usr/bin/env python3
"""
SRA Task Manager Node (v2)

Changes from v1:
- Credentials read from environment variables
- Inventory reservation on task acceptance (prevents oversubscription)
- Internal task queue — one task dispatched at a time, no silent drops
- Atomic DB transactions (autocommit=False, explicit commit/rollback)
- /sra/tasks/status feedback loop to manage queue lifecycle

Bug fixes vs submitted draft:
- release_reservation used DELETE ... LIMIT 1 which PostgreSQL does not support.
  Fixed to use a subquery: DELETE WHERE id = (SELECT id ... LIMIT 1).
- task_status_callback replaced current_active_task with the status dict
  (wrong keys). Fixed to only clear/keep current_active_task without
  overwriting it with incompatible data.
"""

import json
import os
from datetime import datetime, timezone

import psycopg2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ---------------------------------------------------------------------------
# Database configuration from environment
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host":     os.getenv("SRA_DB_HOST",     "localhost"),
    "dbname":   os.getenv("SRA_DB_NAME",     "sra_db"),
    "user":     os.getenv("SRA_DB_USER",     "sra_user"),
    "password": os.getenv("SRA_DB_PASSWORD", ""),
    "port":     int(os.getenv("SRA_DB_PORT", "5432")),
}


class TaskManagerNode(Node):

    def __init__(self):
        super().__init__("sra_task_manager")

        self.conn = self.connect_db()

        # In-memory queue. Tasks wait here until the executor is free.
        # NOTE: queue is lost on node restart. Task recovery (re-queuing
        # tasks with status='queued' from the DB on startup) is a
        # recommended future improvement.
        self.task_queue = []
        self.current_active_task = None  # the task dict currently with executor

        # --- Subscribers ---
        self.sub = self.create_subscription(
            String, "/sra/ai/command", self.command_callback, 10
        )
        self.status_sub = self.create_subscription(
            String, "/sra/tasks/status", self.task_status_callback, 10
        )

        # --- Publishers ---
        self.task_pub  = self.create_publisher(String, "/sra/tasks/active",   10)
        self.alert_pub = self.create_publisher(String, "/sra/alerts/events",  10)

        self.get_logger().info("Task manager ready (reservation + queue enabled).")

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
    # Command handling
    # ------------------------------------------------------------------

    def command_callback(self, msg: String):
        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Malformed JSON on /sra/ai/command")
            return

        if command.get("error"):
            self.get_logger().warn(f'Ignoring failed parse: {command["error"]}')
            return

        config_code = command["config_code"]
        quantity    = command["quantity"]
        destination = command["destination"]

        self.get_logger().info(
            f"Processing: {config_code} x{quantity} → {destination}"
        )

        # Check available inventory (physical stock minus already-reserved units)
        available = self.get_available_inventory(config_code)
        if available is None:
            self.publish_alert("error", f"Config not found in DB: {config_code}")
            return
        if available < quantity:
            self.get_logger().warn(
                f"Insufficient stock for {config_code}: "
                f"requested {quantity}, available {available}"
            )
            self.publish_alert(
                "warning",
                f"Stock insuficiente para {config_code}: "
                f"solicitado {quantity}, disponible {available}"
            )
            return

        # Reserve the units immediately so a second command can't claim the same stock
        if not self.reserve_inventory(config_code, quantity):
            self.publish_alert("error", "No se pudo reservar inventario")
            return

        # Create the persistent task record
        task_id = self.create_task(config_code, quantity, destination)
        if task_id is None:
            self.publish_alert("error", "No se pudo crear la tarea en la base de datos")
            self.release_reservation(config_code, quantity)
            return

        task = {
            "task_id":     task_id,
            "config_code": config_code,
            "quantity":    quantity,
            "destination": destination,
            "status":      "queued",
        }

        self.task_queue.append(task)
        self.get_logger().info(
            f"Task {task_id} queued. Queue length: {len(self.task_queue)}"
        )
        self.publish_alert(
            "info",
            f"Tarea {task_id} en cola para {destination}"
        )

        self.try_dispatch_next_task()

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def try_dispatch_next_task(self):
        """Send the next queued task to the executor if the robot is free."""
        if self.current_active_task is not None or not self.task_queue:
            return

        task = self.task_queue.pop(0)
        self.current_active_task = task

        task_msg = String()
        task_msg.data = json.dumps(task)
        self.task_pub.publish(task_msg)

        self.get_logger().info(
            f"Dispatched task {task['task_id']} to executor"
        )
        self.publish_alert(
            "info",
            f"Tarea {task['task_id']} enviada al robot"
        )

    def task_status_callback(self, msg: String):
        """
        Receive status updates from the executor on /sra/tasks/status.

        Only terminal statuses (completed, failed, cancelled, rejected) free
        the robot for the next task. Intermediate statuses (accepted,
        in_progress) are logged only — current_active_task is NOT replaced
        because the executor's status dict uses different keys than the task dict.
        """
        try:
            status_update = json.loads(msg.data)
            task_id    = status_update.get("task_id")
            new_status = status_update.get("status")

            self.get_logger().info(
                f"Task {task_id} status update: {new_status}"
            )

            if new_status in ("completed", "failed", "cancelled", "rejected"):
                self.current_active_task = None
                self.get_logger().info(
                    f"Task {task_id} finished ({new_status}). "
                    f"Queue length: {len(self.task_queue)}"
                )
                self.try_dispatch_next_task()
            # 'accepted' and 'in_progress' are informational — no queue action needed

        except Exception as exc:
            self.get_logger().error(f"Error processing task status: {exc}")

    # ------------------------------------------------------------------
    # Database operations
    # ------------------------------------------------------------------

    def get_available_inventory(self, config_code: str):
        """
        Available = physical quantity − sum of active reservations.
        Uses LEFT JOIN so configs with no reservations still return their
        full quantity.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT i.quantity - COALESCE(SUM(r.quantity), 0) AS available
                    FROM inventory i
                    JOIN configurations c ON i.config_id = c.id
                    LEFT JOIN inventory_reservations r ON r.config_id = c.id
                    WHERE c.config_code = %s
                    GROUP BY i.id, i.quantity
                """, (config_code,))
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as exc:
            self.get_logger().error(f"Inventory query failed: {exc}")
            return None

    def reserve_inventory(self, config_code: str, quantity: int) -> bool:
        """
        Insert a reservation record so this stock cannot be claimed by another task.
        Rolled back automatically if create_task fails afterwards.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO inventory_reservations (config_id, quantity, created_at)
                    SELECT c.id, %s, NOW()
                    FROM configurations c
                    WHERE c.config_code = %s
                """, (quantity, config_code))
            self.conn.commit()
            return True
        except Exception as exc:
            self.conn.rollback()
            self.get_logger().error(f"Reservation failed: {exc}")
            return False

    def release_reservation(self, config_code: str, quantity: int) -> None:
        """
        Remove the oldest matching reservation for this config/quantity pair.

        FIX: PostgreSQL does not support DELETE ... LIMIT 1 directly.
        A subquery is required to select the target row by its primary key first.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM inventory_reservations
                    WHERE id = (
                        SELECT id FROM inventory_reservations
                        WHERE config_id = (
                            SELECT id FROM configurations WHERE config_code = %s
                        )
                        AND quantity = %s
                        ORDER BY created_at ASC
                        LIMIT 1
                    )
                """, (config_code, quantity))
            self.conn.commit()
        except Exception as exc:
            self.conn.rollback()
            self.get_logger().error(f"Release reservation failed: {exc}")

    def create_task(self, config_code: str, quantity: int, destination: str):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tasks (config_code, quantity, destination, status)
                    VALUES (%s, %s, %s, 'queued')
                    RETURNING id
                """, (config_code, quantity, destination))
                task_id = cur.fetchone()[0]
            self.conn.commit()
            return task_id
        except Exception as exc:
            self.conn.rollback()
            self.get_logger().error(f"Task creation failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TaskManagerNode()
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
