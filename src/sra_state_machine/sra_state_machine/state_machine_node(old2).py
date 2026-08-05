#!/usr/bin/env python3
"""
SRA State Machine Node

Single source of truth for what the SRA system is doing right now.
Observes existing topics (does not modify their producers) and publishes
the current state to /sra/system/state. Enforces timeouts so the system
never hangs silently if a step never completes.
"""

import json
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ----
# Valid states
# ----
STATES = [
    "IDLE", "LISTENING", "PARSING", "RESERVING",
    "DELIVERING", "COMPLETING", "ERROR", "RECOVERING",
]

# Timeout (seconds) allowed in each state before auto-transition to ERROR.
# None = no timeout (IDLE waits forever, ERROR waits for manual/auto recovery).
TIMEOUTS = {
    "LISTENING":  30.0,
    "PARSING":    60.0,
    "RESERVING":  20.0,
    "DELIVERING": 120.0,
    "COMPLETING": 20.0,
    "RECOVERING": 3.0,
}


class StateMachineNode(Node):

    def __init__(self):
        super().__init__("sra_state_machine")

        self.state          = "IDLE"
        self.previous_state = None
        self.task_id         = None
        self.config_code     = None
        self.destination     = None
        self.error_detail     = None

        self.timeout_timer = None

        # --- Publisher ---
        self.state_pub = self.create_publisher(String, "/sra/system/state", 10)

        # --- Subscribers ---
        self.create_subscription(String, "/sra/operator/raw_command", self.on_raw_command, 10)
        self.create_subscription(String, "/sra/ai/parse_result", self.on_ai_command, 10)
        self.create_subscription(String, "/sra/tasks/active", self.on_task_active, 10)
        self.create_subscription(String, "/sra/tasks/status", self.on_task_status, 10)
        self.create_subscription(String, "/sra/alerts/events", self.on_alert, 10)
        self.create_subscription(String, "/sra/system/emergency_stop", self.on_emergency_stop, 10)
        self.create_subscription(String, "/sra/operator/recover", self.on_recover, 10)

        self.tts_pub = self.create_publisher(String, "/sra/tts/speak", 10)

        # Publish the initial state so subscribers see IDLE immediately on startup
        self.publish_state()

        self.get_logger().info("State machine ready. Initial state: IDLE")

    # ----
    # Core transition logic
    # ----

    def transition(self, new_state: str, **context):
        """
        Move to new_state, cancel any pending timeout, arm a new one if
        applicable, update context fields, and publish the new state.
        """
        if new_state not in STATES:
            self.get_logger().error(f"Attempted transition to unknown state: {new_state}")
            return

        if new_state == self.state:
            self.get_logger().warn(f"Attempted transition to the same state you already are: {new_state}")
            return

        self._cancel_timeout()

        self.previous_state = self.state
        self.state = new_state

        # Merge any context fields passed in (task_id, config_code, etc.)
        for key, value in context.items():
            setattr(self, key, value)

        # Clear error_detail whenever we leave ERROR successfully
        if new_state != "ERROR" and self.previous_state == "ERROR":
            self.error_detail = None

        self.get_logger().info(
            f"State: {self.previous_state} → {self.state}"
        )
        self.publish_state()

        timeout = TIMEOUTS.get(new_state)
        if timeout is not None:
            self.timeout_timer = self.create_timer(timeout, self._on_timeout)

    def _on_timeout(self):
        if self.state == "RECOVERING":
            self.get_logger().info("Recovery complete — transitioning to IDLE.")
            self.transition("IDLE")
        else:
            self.get_logger().warn(f"Timeout in state {self.state}")
            self.transition("ERROR", error_detail=f"Timeout in {self.state}")

    def _cancel_timeout(self):
        if self.timeout_timer is not None:
            self.timeout_timer.cancel()
            self.timeout_timer = None

    def publish_state(self):
        payload = {
            "state":          self.state,
            "previous_state": self.previous_state,
            "task_id":        self.task_id,
            "config_code":    self.config_code,
            "destination":    self.destination,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "error_detail":   self.error_detail,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.state_pub.publish(msg)

    # ----
    # Topic callbacks — each one decides if the message is a valid trigger
    # for a transition FROM the current state.
    # ----

    def on_raw_command(self, msg: String):
        """A transcribed command arrived. Valid from IDLE or LISTENING."""
        if self.state in ("IDLE", "LISTENING"):
            self.transition("PARSING")
        else:
            self.get_logger().warn(
                f"Ignoring raw_command while in state {self.state} "
                f"(system busy)"
            )

    def on_ai_command(self, msg: String):
        """LLM parser result arrived. Only meaningful while PARSING."""
        if self.state != "PARSING":
            return
        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError:
            self.transition("ERROR", error_detail="Malformed JSON on /sra/ai/command")
            return

        if command.get("error"):
            self.transition("IDLE", error_detail=None)
        else:
            if command.get("intent") == "query" or command.get("intent") == "status":
                self.transition("IDLE")
            #elif command.get("intent") == "stop":
            #    self.transition("ERROR", error_detail="Se ha solicitado detener el sistema.")
            else:
                self.transition(
                    "RESERVING",
                    config_code=command.get("config_code"),
                    destination=command.get("destination"),
                )
    def on_emergency_stop(self, msg: String) -> None:
        """
        Emergency stop signal. Transitions to ERROR from ANY state.
        No gate — this is the authoritative stop handler.
        """
        self.get_logger().warn(
            f"EMERGENCY STOP received in state {self.state} — forcing ERROR."
        )
        self.transition("ERROR", error_detail="Parada de emergencia activada.")

    def on_recover(self, msg: String) -> None:
        """
        Deliberate operator recovery action. Only valid from ERROR.
        Transitions ERROR → RECOVERING → IDLE.
        RECOVERING is a timed state: after its timeout it auto-advances
        to IDLE, giving TTS time to speak the confirmation.


        Recover must place all the efectors and devices (UR10e and DeliveryBot to home position).--------------------------------------
        """
        if self.state != "ERROR":
            self.get_logger().info(
                f"Recovery request ignored — system is in {self.state}, not ERROR."
            )
            return

        # Speak confirmation so operator knows the system accepted the recovery
        self.transition("RECOVERING")
        tts_msg = String()
        tts_msg.data = "Sistema reiniciado. Listo para recibir órdenes."
        self.tts_pub.publish(tts_msg)

        self.get_logger().info("Recovery requested by operator — transitioning to RECOVERING.")

    def on_task_active(self, msg: String):
        """Task manager dispatched a task to the executor. Only from RESERVING."""
        if self.state != "RESERVING":
            return
        try:
            task = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.transition(
            "DELIVERING",
            task_id=task.get("task_id"),
            config_code=task.get("config_code"),
            destination=task.get("destination"),
        )

    def on_task_status(self, msg: String):
        """Executor lifecycle updates. completed/failed only matter while DELIVERING."""
        if self.state != "DELIVERING":
            return
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        new_status = status.get("status")
        if new_status == "completed":
            self.transition("COMPLETING")
            # DB is already updated by delivery_sim before this message is sent,
            # so COMPLETING is momentary — go straight to IDLE.
            self.transition("IDLE")
        elif new_status == "failed":
            self.transition("ERROR", error_detail=f"Task {status.get('task_id')} failed")
        elif new_status == "rejected":
            self.transition("ERROR", error_detail="Executor rejected task while busy")
        # 'accepted' / 'in_progress' — no transition needed, already DELIVERING

    def on_alert(self, msg: String):
        """System-wide alerts. Only react to hard errors here."""
        try:
            alert = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if alert.get("level") == "error" and self.state != "ERROR":
            self.transition("ERROR", error_detail=alert.get("message"))


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
