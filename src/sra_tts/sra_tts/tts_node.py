#!/usr/bin/env python3
"""
SRA TTS Node

Subscribes to:
  - /sra/tts/speak       (String, plain text) — direct speech requests only

Synthesizes speech locally with Piper (offline) and plays it through
the industrial speaker via aplay. All work happens on a background
worker thread so ROS callbacks never block.
"""

import os
import queue
import subprocess
import tempfile
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

VOICES_DIR = os.getenv("SRA_PIPER_VOICES_DIR", "")
ES_MODEL   = os.getenv("SRA_PIPER_ES_MODEL", "es_ES-sharvard-medium.onnx")


class TTSNode(Node):

    def __init__(self):
        super().__init__("sra_tts_node")

        self.speak_queue: "queue.Queue[str]" = queue.Queue()

        self.create_subscription(
            String, "/sra/tts/speak", self.on_speak, 10
        )

        self.speaking_pub = self.create_publisher(Bool,"/sra/tts/speaking",10,)
        
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        self.get_logger().info(
            f"TTS node ready. Voices dir: {VOICES_DIR}"
        )

    # ----
    # Callbacks — fast, just enqueue
    # ----

    def on_speak(self, msg: String):
        text = msg.data.strip()
        if text:
            self.speak_queue.put(text)

    # ----
    # Worker thread — does the actual synthesis + playback
    # ----

    def _worker_loop(self):
        while rclpy.ok():     # type: ignore[attr-defined]
            try:
                text = self.speak_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._speak(text)

    def _speak(self, text: str):
        model_path = os.path.join(VOICES_DIR, ES_MODEL)

        if not os.path.exists(model_path):
            self.get_logger().error(f"Piper model not found: {model_path}")
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            # Piper reads text from stdin, writes a wav file
            subprocess.run(
                ["piper", "--model", model_path, "--output_file", wav_path],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=15,
            )

            self._publish_speaking(True)
            try:
                subprocess.run(
                    ["aplay", wav_path],
                    check=True,
                )
            finally:
                self._publish_speaking(False)

        except subprocess.CalledProcessError as exc:
            self.get_logger().error(f"Piper/aplay failed: {exc}")
        except subprocess.TimeoutExpired:
            self.get_logger().error("TTS synthesis/playback timed out")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def _publish_speaking(self, speaking: bool) -> None:
        """Publish the current speaker playback state."""
        msg = Bool()
        msg.data = speaking
        self.speaking_pub.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
