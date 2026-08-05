#!/usr/bin/env python3
"""
SRA Task Manager Node (v3)

Changes from v2:
  Bug #2 fix — failed/rejected/cancelled tasks now release their inventory
               reservation and write a terminal status to the DB.
  Bug #3 fix — reservations are scoped to task_id, eliminating the
               oldest-match heuristic that could free the wrong reservation
               when two tasks share the same config_code and quantity.

New task creation flow (required by Bug #3):
  1. INSERT task row with status='pending_reservation'  → get task_id
  2. INSERT reservation tagged with that task_id
  3. UPDATE task row to status='queued'
  If step 2 or 3 fails, the task row is deleted and the reservation
  (if it was inserted) is released by task_id.
"""

import json
import os
from datetime import datetime, timezone

import psycopg2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

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

        self.conn = self._connect_db()

        self.task_queue          = []
        self.current_active_task = None  # full task dict currently with executor

        # Subscribers
        self.create_subscription(
            String, "/sra/ai/command",    self.command_callback,     10
        )
        self.create_subscription(
            String, "/sra/tasks/status",  self.task_status_callback, 10
        )
        
        self.create_subscription(
            String, "/sra/system/emergency_stop", self.emergency_stop_callback, 10
        )

        # Publishers
        self.task_pub  = self.create_publisher(String, "/sra/tasks/active",  10)
        self.alert_pub = self.create_publisher(String, "/sra/alerts/events", 10)

        self.get_logger().info("Task manager v3 ready.")

    # ──────────────────────────────────────────────
    # DB connection
    # ──────────────────────────────────────────────

    def _connect_db(self):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = False
            self.get_logger().info("Database connected.")
            return conn
        except Exception as exc:
            self.get_logger().error(f"Database connection failed: {exc}")
            raise RuntimeError(f"Cannot start without database: {exc}")

    # ──────────────────────────────────────────────
    # Command handling
    # ──────────────────────────────────────────────

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
        available = self._get_available_inventory(config_code)
        if available is None:
            self._publish_alert("error", f"Config not found in DB: {config_code}")
            return
        if available < quantity:
            self._publish_alert(
                "warning",
                f"Stock insuficiente para {config_code}: "
                f"solicitado {quantity}, disponible {available}"
            )
            return

        # Create task + reservation atomically (Bug #3 ordering)
        task_id = self._create_task_with_reservation(
            config_code, quantity, destination
        )
        if task_id is None:
            self._publish_alert("error", "No se pudo crear la tarea o reservar inventario")
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
        self._publish_alert("info", f"Tarea {task_id} en cola para {destination}")
        self._try_dispatch_next_task()

    # ──────────────────────────────────────────────
    # Queue management
    # ──────────────────────────────────────────────

    def _try_dispatch_next_task(self):
        if self.current_active_task is not None or not self.task_queue:
            return

        task = self.task_queue.pop(0)
        self.current_active_task = task

        out = String()
        out.data = json.dumps(task)
        self.task_pub.publish(out)

        self.get_logger().info(f"Dispatched task {task['task_id']} to executor")
        self._publish_alert("info", f"Tarea {task['task_id']} enviada al robot")

    def task_status_callback(self, msg: String):
        try:
            update     = json.loads(msg.data)
            task_id    = update.get("task_id")
            new_status = update.get("status")

            self.get_logger().info(f"Task {task_id} status: {new_status}")

            if new_status == "completed":
                # delivery_sim already wrote the terminal DB status and
                # released the reservation inside finalize_task_in_db().
                # We only need to free the executor slot here.
                self.current_active_task = None
                self._try_dispatch_next_task()

            elif new_status in ("failed", "cancelled", "rejected"):
                # Bug #2 fix: release reservation and write terminal DB status.
                # We use the task_id from the status update, not from
                # current_active_task, so this is safe even if the active
                # task reference is stale.
                self._release_reservation_by_task(task_id)
                self._write_terminal_status(task_id, new_status)

                self.current_active_task = None
                self.get_logger().info(
                    f"Task {task_id} ended ({new_status}). "
                    f"Reservation released. Queue: {len(self.task_queue)}"
                )
                self._publish_alert(
                    "warning",
                    f"Tarea {task_id} terminó con estado: {new_status}"
                )
                self._try_dispatch_next_task()

            # 'accepted' and 'in_progress' are informational — no action needed

        except Exception as exc:
            self.get_logger().error(f"Error processing task status: {exc}")
    
    
    def emergency_stop_callback(self, msg: String) -> None:
        """
        On emergency stop: drain the queue and release the reservation
        for the currently active task. The active task itself is being
        cancelled by delivery_sim — task_manager only cleans up its own state.
        """
        self.get_logger().warn("EMERGENCY STOP received — clearing task queue.")

        # Release reservation for the in-flight task (Bug #3 fix makes this safe)
        if self.current_active_task is not None:
            task_id = self.current_active_task["task_id"]
            self._release_reservation_by_task(task_id)
            self._write_terminal_status(task_id, "cancelled")
            self.get_logger().warn(
                f"Task {task_id} cancelled. Reservation released."
            )

        # Release reservations for every queued task
        for task in self.task_queue:
            self._release_reservation_by_task(task["task_id"])
            self._write_terminal_status(task["task_id"], "cancelled")
            self.get_logger().warn(
                f"Queued task {task['task_id']} cancelled."
            )

        self.task_queue.clear()
        self.current_active_task = None

        self._publish_alert("error", "Cola de tareas vaciada por parada de emergencia.")
        

    # ──────────────────────────────────────────────
    # DB operations
    # ──────────────────────────────────────────────

    def _get_available_inventory(self, config_code: str):
        """Physical stock minus sum of all active reservations for this config."""
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

    def _create_task_with_reservation(
        self, config_code: str, quantity: int, destination: str
    ):
        """
        Bug #3 fix: atomic three-step flow.

          1. INSERT task with status='pending_reservation' → get task_id
          2. INSERT reservation with that task_id
          3. UPDATE task to status='queued'

        Any failure rolls back the entire transaction so no orphaned rows
        or reservations are left behind.
        """
        try:
            with self.conn.cursor() as cur:

                # Step 1 — create task row, get its id
                cur.execute("""
                    INSERT INTO tasks (config_code, quantity, destination, status)
                    VALUES (%s, %s, %s, 'pending_reservation')
                    RETURNING id
                """, (config_code, quantity, destination))
                task_id = cur.fetchone()[0]

                # Step 2 — reserve inventory, tagged with task_id
                cur.execute("""
                    INSERT INTO inventory_reservations
                        (config_id, quantity, task_id, created_at)
                    SELECT c.id, %s, %s, NOW()
                    FROM configurations c
                    WHERE c.config_code = %s
                """, (quantity, task_id, config_code))

                # Step 3 — promote task to queued
                cur.execute("""
                    UPDATE tasks SET status = 'queued' WHERE id = %s
                """, (task_id,))

            self.conn.commit()
            self.get_logger().info(
                f"Task {task_id} created with scoped reservation."
            )
            return task_id

        except Exception as exc:
            self.conn.rollback()
            self.get_logger().error(f"Task+reservation creation failed: {exc}")
            return None

    def _release_reservation_by_task(self, task_id: int) -> None:
        """
        Bug #3 fix: delete the reservation that belongs to this specific task.
        No heuristic — exact match on task_id.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM inventory_reservations WHERE task_id = %s
                """, (task_id,))
            self.conn.commit()
            self.get_logger().info(
                f"Reservation released for task {task_id}."
            )
        except Exception as exc:
            self.conn.rollback()
            self.get_logger().error(
                f"Failed to release reservation for task {task_id}: {exc}"
            )

    def _write_terminal_status(self, task_id: int, status: str) -> None:
        """
        Bug #2 fix: write the terminal status to the tasks table so the DB
        row never stays permanently at 'queued' or 'in_progress'.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE tasks
                    SET status = %s, completed_at = NOW()
                    WHERE id = %s
                """, (status, task_id))
            self.conn.commit()
        except Exception as exc:
            self.conn.rollback()
            self.get_logger().error(
                f"Failed to write terminal status for task {task_id}: {exc}"
            )

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _publish_alert(self, level: str, message: str) -> None:
        alert = {
            "level":     level,
            "message":   message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        out = String()
        out.data = json.dumps(alert)
        self.alert_pub.publish(out)

    def destroy_node(self):
        if self.conn:
            self.conn.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TaskManagerNode()
    except RuntimeError as exc:
        print(f"[FATAL] {exc}")
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
