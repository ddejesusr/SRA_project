#!/usr/bin/env python3
"""
SRA Voice Input Node

Captures microphone audio, uses WebRTC VAD to isolate an operator utterance,
transcribes it with Faster-Whisper on the CPU, rejects low-confidence output,
and publishes accepted text to /sra/operator/raw_command.
"""

import collections
import os
import tempfile
import threading
import wave

import pyaudio
import rclpy
import webrtcvad
from faster_whisper import WhisperModel
from rclpy.node import Node
from std_msgs.msg import String


# --- Audio capture settings ---
SAMPLE_RATE = 16000              # Hz; supported by WebRTC VAD and Whisper
FRAME_DURATION = 30              # ms; WebRTC VAD accepts 10, 20, or 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
CHANNELS = 1
FORMAT = pyaudio.paInt16         # 16-bit PCM

# --- Utterance detection settings ---
PRE_BUFFER_FRAMES = 10           # 300 ms of audio before speech is confirmed
START_TRIGGER_FRAMES = 3         # 90 ms of consecutive speech to start recording
SILENCE_THRESHOLD_FRAMES = 30    # 900 ms of silence ends an utterance
END_PADDING_FRAMES = 8           # retain 240 ms after the final voiced frame
MIN_VOICED_FRAMES = 15           # 450 ms of confirmed speech required

# --- Faster-Whisper settings ---
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_LANGUAGE = "es"
BEAM_SIZE = 3

# Reject uncertain decoded segments before publishing them to the command parser.
# These are starting values; tune them with recordings from the real work area.
MAX_NO_SPEECH_PROB = 0.60
MIN_AVG_LOGPROB = -1.20
MAX_COMPRESSION_RATIO = 2.40

# Domain vocabulary helps Whisper transcribe technical terms
# correctly. Do not include hallucinated phrases here — use HALLUCINATION_BLOCKLIST.
INITIAL_PROMPT = (
    #"Sistema industrial de ensamblaje de cajas de fusibles. "
    "Vocabulario: fusible, fusibles, caja de fusibles, "
    "estación A, estación B, inventario, producción, entrega, "
    "robot, cobot, UR10e, azul, rojo, negro, ambos, superior, inferior."
)

# Known Whisper hallucinations on silence and noise.
# These phrases are generated with high confidence from training data, so
# confidence-score filters alone do not reliably block them.
# Add any new hallucinations observed in the real deployment environment.
# Comparison is case-insensitive and uses substring matching.
HALLUCINATION_BLOCKLIST = [
    "subtítulos por la comunidad de amara",
    "suscríbete",
    "suscribete",
    "hasta la próxima",
    "hasta la proxima",
    "gracias por ver",
    "no olvides suscribirte",
    "like y suscríbete",
    "amara.org",
]


class VoiceInputNode(Node):
    """Captures, transcribes, filters, and publishes operator voice commands."""

    def __init__(self):
        super().__init__("sra_voice_input")

        self.pub = self.create_publisher(
            String,
            "/sra/operator/raw_command",
            10,
        )

        self.get_logger().info(
            f"Loading Faster-Whisper ({WHISPER_MODEL_SIZE}, "
            f"{WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE})..."
        )
        self.whisper = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        self.get_logger().info("Whisper model loaded.")

        # 0 is least aggressive; 3 filters most non-speech audio.
        self.vad = webrtcvad.Vad(3)
        self.audio = pyaudio.PyAudio()

        self.running = True
        self.listen_thread = threading.Thread(
            target=self.listen_loop,
            daemon=True,
        )
        self.listen_thread.start()

        self.get_logger().info("Voice input ready. Listening for commands...")

    def listen_loop(self):
        """Continuously capture audio and send completed, valid utterances to STT."""
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE,
        )

        pre_buffer = collections.deque(maxlen=PRE_BUFFER_FRAMES)
        utterance_frames = []
        triggered = False
        silent_count = 0
        voiced_count = 0
        last_voiced_frame_index = -1
        speech_start_count = 0

        self.get_logger().info("Microphone stream open.")

        try:
            while self.running:
                try:
                    raw_frame = stream.read(
                        FRAME_SIZE,
                        exception_on_overflow=False,
                    )
                except Exception as exc:
                    self.get_logger().warning(f"Audio read error: {exc}")
                    continue

                try:
                    is_speech = self.vad.is_speech(raw_frame, SAMPLE_RATE)
                except Exception:
                    is_speech = False

                if not triggered:
                    # Preserve a short pre-roll so the first syllable is not lost.
                    pre_buffer.append(raw_frame)
                    speech_start_count = (
                        speech_start_count + 1 if is_speech else 0
                    )

                    # A single transient sound must not open an utterance.
                    if speech_start_count >= START_TRIGGER_FRAMES:
                        triggered = True
                        silent_count = 0
                        utterance_frames = list(pre_buffer)
                        voiced_count = speech_start_count
                        last_voiced_frame_index = len(utterance_frames) - 1
                        self.get_logger().info("Speech detected, recording...")
                    continue

                # An active utterance includes speech and the short pauses within it.
                utterance_frames.append(raw_frame)
                if is_speech:
                    voiced_count += 1
                    silent_count = 0
                    last_voiced_frame_index = len(utterance_frames) - 1
                else:
                    silent_count += 1

                if silent_count < SILENCE_THRESHOLD_FRAMES:
                    continue

                # Do not send the full end-of-utterance silence/noise to Whisper.
                end_index = min(
                    len(utterance_frames),
                    last_voiced_frame_index + 1 + END_PADDING_FRAMES,
                )
                trimmed_frames = utterance_frames[:end_index]

                self.get_logger().info(
                    "Speech ended: %d voiced frames; %d total frames after trimming."
                    % (voiced_count, len(trimmed_frames))
                )

                if voiced_count >= MIN_VOICED_FRAMES:
                    self.transcribe_and_publish(trimmed_frames)
                else:
                    self.get_logger().info(
                        "Too little confirmed speech; ignoring utterance."
                    )

                # Reset state for the next utterance.
                utterance_frames = []
                triggered = False
                silent_count = 0
                voiced_count = 0
                last_voiced_frame_index = -1
                speech_start_count = 0
                pre_buffer.clear()
        finally:
            stream.stop_stream()
            stream.close()

    def _is_hallucination(self, text: str) -> bool:
        """
        Return True if the text matches a known Whisper hallucination phrase.

        Confidence-score filters do not reliably block phrases that appear
        frequently in Whisper training data. An explicit blocklist is more
        deterministic for known offenders.
        """
        text_lower = text.lower()
        for phrase in HALLUCINATION_BLOCKLIST:
            if phrase in text_lower:
                self.get_logger().warning(
                    f"Known hallucination blocked: {text!r}"
                )
                return True
        return False

    def transcribe_and_publish(self, frames):
        """Write an utterance to WAV, transcribe it, filter it, and publish it."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            with wave.open(tmp_path, "wb") as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(self.audio.get_sample_size(FORMAT))
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(b"".join(frames))

            segments, info = self.whisper.transcribe(
                tmp_path,
                language=WHISPER_LANGUAGE,
                beam_size=BEAM_SIZE,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400},
                initial_prompt=INITIAL_PROMPT,
            )

            accepted_segments = []
            for segment in segments:
                text = segment.text.strip()
                self.get_logger().info(
                    "STT segment: text=%r, no_speech_prob=%.2f, "
                    "avg_logprob=%.2f, compression_ratio=%.2f"
                    % (
                        text,
                        segment.no_speech_prob,
                        segment.avg_logprob,
                        segment.compression_ratio,
                    )
                )

                # Layer 1: confidence metrics
                is_reliable = (
                    bool(text)
                    and segment.no_speech_prob <= MAX_NO_SPEECH_PROB
                    and segment.avg_logprob >= MIN_AVG_LOGPROB
                    and segment.compression_ratio <= MAX_COMPRESSION_RATIO
                )

                if not is_reliable:
                    self.get_logger().warning(
                        f"Rejected uncertain transcription: {text!r}"
                    )
                    continue

                # Layer 2: known hallucination blocklist
                if self._is_hallucination(text):
                    continue

                accepted_segments.append(text)

            text = " ".join(accepted_segments).strip()
            if not text:
                self.get_logger().info(
                    "No reliable transcription produced; ignoring utterance."
                )
                return

            msg = String()
            msg.data = text
            self.pub.publish(msg)
            self.get_logger().info(
                f"Published accepted transcription [{info.language}]: {text!r}"
            )

        except Exception as exc:
            self.get_logger().error(f"Transcription failed: {exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def destroy_node(self):
        self.running = False
        if self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2.0)
        self.audio.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoiceInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
