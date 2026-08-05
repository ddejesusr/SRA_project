#!/usr/bin/env python3
"""
SRA LLM Command Parser Node (v2)

Changes from v1:
- Credentials and model config read from environment variables
- Maximum delivery quantity enforced (SRA_MAX_DELIVERY_QUANTITY)
- Parse failures published as alerts to /sra/alerts/events
- Stricter SYSTEM_PROMPT: LLM must reject non-delivery inputs explicitly
"""

import json
import os
from datetime import datetime, timezone

import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ---------------------------------------------------------------------------
# Configuration — all values read from environment with safe defaults.
# Set these by sourcing your .env file before launching the node.
# ---------------------------------------------------------------------------
OLLAMA_URL   = os.getenv("SRA_OLLAMA_URL",   "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("SRA_OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
MAX_QUANTITY = int(os.getenv("SRA_MAX_DELIVERY_QUANTITY", "6"))

VALID_BACK_COLORS = {"blue", "red", "black"}
VALID_FRONT_COLORS = {"blue", "red"}
VALID_FUSE_POS = {"none", "upper", "lower", "both"}
VALID_STATIONS = {"station_A", "station_B"}

# ---------------------------------------------------------------------------
# Strict system prompt.
# The LLM must refuse inputs that are not clearly delivery instructions.
# This is the second line of defence after the voice node's VAD/confidence
# filters. It prevents hallucinated transcriptions that slipped through from
# generating real database entries.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a command parser for an industrial robot assistant that manages fuse box delivery.

The robot delivers fuse boxes to workstations. Each fuse box has a configuration code:
  back_color-front_color-fuse_position
  back_color:    blue, red, black
  front_color:   blue, red
  fuse_position: none, upper, lower, both

Valid destinations: station_A, station_B

CRITICAL DISAMBIGUATION — QUANTITY vs FUSE POSITION:
The word "dos" (two) or "two" in a command almost always refers to the NUMBER OF BOXES,
NOT the fuse position. The fuse position is described by WHERE the fuses are, not how many.

Fuse position Spanish mappings:
- "sin fusibles", "ninguno", "sin fusible"          → fuse_position = none
- "fusible superior", "arriba", "solo arriba"       → fuse_position = upper
- "fusible inferior", "abajo", "solo abajo"         → fuse_position = lower
- "ambos fusibles", "los dos fusibles", "arriba y abajo",
  "superior e inferior", "dos fusibles", "con fusibles" → fuse_position = both

EXAMPLES (study these carefully):
  "una caja azul y roja de dos fusibles a la estación A"
   → config: blue-red-both, quantity: 1, destination: station_A
   ("dos fusibles" = both fuse positions, "una caja" = 1 box)

  "dos cajas azul-rojo-ambos a la estación B"
   → config: blue-red-both, quantity: 2, destination: station_B
   ("dos cajas" = 2 boxes, "ambos" = both fuse positions)

  "envíame tres cajas negras y azules sin fusibles a la estación A"
   → config: black-blue-none, quantity: 3, destination: station_A

  "una caja roja y azul con fusible superior a la estación B"
   → config: red-blue-upper, quantity: 1, destination: station_B

STRICT RULES:
- Only parse messages that are clearly delivery instructions from an operator.
- A valid command MUST include both a configuration AND a destination.
- Do NOT guess or invent values not clearly stated.
- If any required field is ambiguous or missing, return an error.

Respond ONLY with a valid JSON object. No explanation, no extra text, no markdown.

JSON format:
{
  "config_code": "back_color-front_color-fuse_position",
  "quantity": <integer>,
  "destination": "station_A" or "station_B",
  "confidence": <float 0.0 to 1.0>,
  "error": null or "<short reason if not a valid delivery command>"
}
"""


class CommandParserNode(Node):

    def __init__(self):
        super().__init__("sra_command_parser")

        self.model        = OLLAMA_MODEL
        self.url          = OLLAMA_URL
        self.max_quantity = MAX_QUANTITY

        # Subscriber: raw text from voice input or direct text publisher
        self.sub = self.create_subscription(
            String, "/sra/operator/raw_command", self.command_callback, 10
        )

        # Publisher: validated structured command for task manager
        self.pub = self.create_publisher(String, "/sra/ai/command", 10)

        # Publisher: operator-facing alerts (parse errors, warnings)
        self.alert_pub = self.create_publisher(String, "/sra/alerts/events", 10)

        self.get_logger().info(
            f"Command parser ready. Model: {self.model}  MaxQty: {self.max_quantity}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def command_callback(self, msg: String):
        raw_text = msg.data
        self.get_logger().info(f'Received command: "{raw_text}"')

        parsed = self.parse_command(raw_text)

        result_msg = String()
        result_msg.data = json.dumps(parsed, ensure_ascii=False)
        self.pub.publish(result_msg)

        if parsed.get("error"):
            self.get_logger().warn(f'Parse error: {parsed["error"]}')
            self._publish_alert(
                "warning",
                f'No pude interpretar el comando: "{raw_text}"'
            )
        else:
            self.get_logger().info(
                f'Parsed → {parsed["config_code"]} '
                f'x{parsed["quantity"]} → {parsed["destination"]}'
            )

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def parse_command(self, text: str) -> dict:
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": text},
                    ],
                    "stream": False,
                },
                timeout=25,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"].strip()

            # Strip markdown code fences if the LLM wraps its response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            parsed = json.loads(content)
            return self.validate(parsed)

        except requests.exceptions.Timeout:
            return self.error_result("Ollama request timed out")
        except json.JSONDecodeError as exc:
            return self.error_result(f"LLM returned invalid JSON: {exc}")
        except Exception as exc:
            return self.error_result(f"Unexpected error: {exc}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, parsed: dict) -> dict:
        """Validate LLM output against known-good values for every field."""

        # If the LLM itself flagged an error, pass it through unchanged
        if parsed.get("error"):
            return parsed

        code = parsed.get("config_code", "")
        qty  = parsed.get("quantity",    0)
        dest = parsed.get("destination", "")
        conf = parsed.get("confidence",  0.0)

        # Config code must be exactly three dash-separated parts
        parts = str(code).split("-")
        if len(parts) != 3:
            return self.error_result(f"Invalid config_code format: {code}")

        back, front, fuse = parts
        if back  not in VALID_BACK_COLORS:
            return self.error_result(f"Unknown back color: {back}")
        if front not in VALID_FRONT_COLORS:
            return self.error_result(f"Unknown front color: {front}")
        if fuse  not in VALID_FUSE_POS:
            return self.error_result(f"Unknown fuse position: {fuse}")
        if dest  not in VALID_STATIONS:
            return self.error_result(f"Unknown destination: {dest}")

        # Quantity must be a positive integer within the physical capacity limit
        if not isinstance(qty, int) or qty < 1 or qty > self.max_quantity:
            return self.error_result(
                f"Invalid quantity: {qty} "
                f"(must be 1–{self.max_quantity})"
            )

        # Confidence is diagnostic only — log it but never block a valid command
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            parsed["confidence"] = 0.0

        return parsed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_alert(self, level: str, message: str) -> None:
        alert = {
            "level":     level,
            "message":   message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        msg = String()
        msg.data = json.dumps(alert)
        self.alert_pub.publish(msg)

    @staticmethod
    def error_result(reason: str) -> dict:
        return {
            "config_code":  None,
            "quantity":     None,
            "destination":  None,
            "confidence":   0.0,
            "error":        reason,
        }


def main(args=None):
    rclpy.init(args=args)
    node = CommandParserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
