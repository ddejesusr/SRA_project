#!/usr/bin/env python3
"""
SRA LLM Command Parser Node
Receives operator text commands, parses them via Ollama,
and publishes structured delivery commands to ROS 2.
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests

# Valid values the system accepts
VALID_BACK_COLORS   = {"blue", "red", "black"}
VALID_FRONT_COLORS  = {"blue", "red"}
VALID_FUSE_POS      = {"none", "upper", "lower", "both"}
VALID_STATIONS      = {"station_A", "station_B"}

SYSTEM_PROMPT = """
You are a command parser for an industrial robot assistant.
The robot manages fuse boxes with 24 configurations.
Each configuration is coded as: back_color-front_color-fuse_position
  back_color:    blue, red, black
  front_color:   blue, red
  fuse_position: none, upper, lower, both

The robot delivers boxes to station_A or station_B.

Your job: extract the delivery intent from the operator's message.
Respond ONLY with a valid JSON object. No explanation, no extra text.

JSON format:
{
  "config_code": "back_color-front_color-fuse_position",
  "quantity": <integer>,
  "destination": "station_A" or "station_B",
  "confidence": <float 0.0 to 1.0>,
  "error": null or "<reason if you could not parse>"
}

If you cannot parse a valid command, set error to a short reason and
set config_code, quantity, destination to null.
"""


class CommandParserNode(Node):

    def __init__(self):
        super().__init__('sra_command_parser')

        # Parameter: Ollama model name (changeable without recompiling)
        self.declare_parameter('ollama_model', 'llama3.1:8b-instruct-q4_K_M')
        self.declare_parameter('ollama_url',   'http://localhost:11434/api/chat')

        self.model = self.get_parameter('ollama_model').get_parameter_value().string_value
        self.url   = self.get_parameter('ollama_url').get_parameter_value().string_value

        # Subscriber: raw operator text input
        self.sub = self.create_subscription(
            String,
            '/sra/operator/raw_command',
            self.command_callback,
            10
        )

        # Publisher: structured parsed command
        self.pub = self.create_publisher(
            String,
            '/sra/ai/command',
            10
        )

        self.get_logger().info(f'Command parser ready. Model: {self.model}')

    def command_callback(self, msg: String):
        raw_text = msg.data
        self.get_logger().info(f'Received command: "{raw_text}"')

        parsed = self.parse_command(raw_text)
        result_msg = String()
        result_msg.data = json.dumps(parsed, ensure_ascii=False)
        self.pub.publish(result_msg)

        if parsed.get('error'):
            self.get_logger().warn(f'Parse error: {parsed["error"]}')
        else:
            self.get_logger().info(
                f'Parsed → {parsed["config_code"]} x{parsed["quantity"]} → {parsed["destination"]}'
            )

    def parse_command(self, text: str) -> dict:
        try:
            response = requests.post(self.url, json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text}
                ],
                "stream": False
            }, timeout=30)

            response.raise_for_status()
            content = response.json()["message"]["content"].strip()

            # Strip markdown code fences if the LLM adds them
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            parsed = json.loads(content)
            return self.validate(parsed)

        except requests.exceptions.Timeout:
            return self.error_result("Ollama request timed out")
        except json.JSONDecodeError as e:
            return self.error_result(f"LLM returned invalid JSON: {e}")
        except Exception as e:
            return self.error_result(f"Unexpected error: {e}")

    def validate(self, parsed: dict) -> dict:
        """Validate the LLM output against known valid values."""
        if parsed.get('error'):
            return parsed

        code  = parsed.get('config_code', '')
        qty   = parsed.get('quantity', 0)
        dest  = parsed.get('destination', '')

        parts = str(code).split('-')
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
        if not isinstance(qty, int) or qty < 1:
            return self.error_result(f"Invalid quantity: {qty}")

        return parsed

    @staticmethod
    def error_result(reason: str) -> dict:
        return {
            "config_code":  None,
            "quantity":     None,
            "destination":  None,
            "confidence":   0.0,
            "error":        reason
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


if __name__ == '__main__':
    main()
