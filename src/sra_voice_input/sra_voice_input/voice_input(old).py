#!/usr/bin/env python3
"""
SRA Voice Input Node

Listens to the microphone, detects when someone speaks,
transcribes the audio using Faster-Whisper on the GPU,
and publishes the text to /sra/operator/raw_command.
"""

import collections
import threading
import struct
import wave
import tempfile
import os

import pyaudio
import webrtcvad
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from faster_whisper import WhisperModel


# --- Audio capture settings ---
SAMPLE_RATE    = 16000   # Hz — required by WebRTC VAD and Whisper
FRAME_DURATION = 30      # ms per audio frame (10, 20, or 30 — VAD requirement)
FRAME_SIZE     = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # samples per frame
CHANNELS       = 1       # mono
FORMAT         = pyaudio.paInt16  # 16-bit audio

# How many consecutive silent frames before we decide speech has ended
SILENCE_THRESHOLD_FRAMES = 30  # 30 × 30ms = 900ms of silence = end of utterance

# Minimum number of voiced frames to bother transcribing
# (avoids transcribing a cough or brief noise)
#MIN_VOICED_FRAMES = 10  # 10 x 30ms = 300ms minumum speech 
MIN_VOICED_FRAMES = 30  # 30 × 30ms = 900ms minimum speech

# Whisper model settings
#WHISPER_MODEL_SIZE = "base"   # base = good balance of speed vs accuracy
WHISPER_MODEL_SIZE = "small" # small = more accurate but a bit slower
#WHISPER_DEVICE     = "cuda"   # use GPU
WHISPER_DEVICE     = "cpu"      # CPU - Fast enough for short commands, keeps GPU free for LLM 
WHISPER_LANGUAGE   = None     # None = auto-detect (handles Spanish and English)
WHISPER_LANGUAGE   = "es"     # force spanish languaje - eliminates languaje confusion


class VoiceInputNode(Node):

    def __init__(self):
        super().__init__('sra_voice_input')

        # Publisher: sends transcribed text to the command parser
        self.pub = self.create_publisher(
            String,
            '/sra/operator/raw_command',
            10
        )

        # Load Whisper model onto GPU
        self.get_logger().info(
            f'Loading Faster-Whisper ({WHISPER_MODEL_SIZE})...'
        )
        self.whisper = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            #compute_type="float16"  # float16 is faster on GPU, accurate enough
	    compute_type="int8"     #int8 is fastest compute type on CPU
        )
        self.get_logger().info('Whisper model loaded.')

        # Voice Activity Detector
        # Aggressiveness 2 = moderate filtering (0=least, 3=most aggressive)
        self.vad = webrtcvad.Vad(3)

        # PyAudio instance
        self.audio = pyaudio.PyAudio()

        # Run microphone capture in a background thread
        # so ROS 2 spin() stays responsive
        self.running = True
        self.listen_thread = threading.Thread(
            target=self.listen_loop,
            daemon=True
        )
        self.listen_thread.start()

        self.get_logger().info(
            'Voice input ready. Listening for commands...'
        )

    def listen_loop(self):
        """
        Runs in a background thread.
        Continuously reads microphone frames, detects speech,
        and triggers transcription when an utterance ends.
        """
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE
        )

        # Ring buffer: keeps the last N frames before speech starts
        # This captures the beginning of a word that started before
        # the VAD detected speech
        pre_buffer = collections.deque(maxlen=10)

        voiced_frames  = []   # frames that contain speech
        triggered      = False  # are we currently in a speech segment?
        silent_count   = 0    # consecutive silent frames since speech ended

        self.get_logger().info('Microphone stream open.')

        while self.running:
            try:
                raw_frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            except Exception as e:
                self.get_logger().warn(f'Audio read error: {e}')
                continue

            # Ask VAD: is this frame voiced (True) or silent (False)?
            try:
                is_speech = self.vad.is_speech(raw_frame, SAMPLE_RATE)
            except Exception:
                is_speech = False

            if not triggered:
                # Not recording yet — buffer frames and watch for speech
                pre_buffer.append(raw_frame)
                if is_speech:
                    # Speech detected — start recording
                    triggered = True
                    silent_count = 0
                    # Include pre-buffer so we don't miss the first syllable
                    voiced_frames = list(pre_buffer)
                    self.get_logger().info('Speech detected, recording...')
            else:
                # Currently recording — keep collecting frames
                voiced_frames.append(raw_frame)

                if not is_speech:
                    silent_count += 1
                else:
                    silent_count = 0  # reset silence counter on any speech

                # End of utterance: enough silence after speech
                if silent_count > SILENCE_THRESHOLD_FRAMES:
                    self.get_logger().info(
                        f'Speech ended ({len(voiced_frames)} frames). Transcribing...'
                    )

                    if len(voiced_frames) >= MIN_VOICED_FRAMES:
                        self.transcribe_and_publish(voiced_frames)
                    else:
                        self.get_logger().info('Too short — ignoring.')

                    # Reset for next utterance
                    voiced_frames = []
                    triggered     = False
                    silent_count  = 0
                    pre_buffer.clear()

        stream.stop_stream()
        stream.close()

    def transcribe_and_publish(self, frames):
        """
        Saves recorded frames to a temp WAV file,
        runs Faster-Whisper transcription,
        and publishes the result.
        """
        # Write frames to a temporary WAV file
        # (Faster-Whisper needs a file, not raw bytes)
        with tempfile.NamedTemporaryFile(
            suffix='.wav', delete=False
        ) as tmp:
            tmp_path = tmp.name

        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))

        try:
            segments, info = self.whisper.transcribe(
                tmp_path,
                beam_size=5,
                language=WHISPER_LANGUAGE,
                condition_on_previous_text=False
            )

            # Collect all segments into one string
            text = ' '.join(seg.text.strip() for seg in segments).strip()

            self.get_logger().info(
                f'Transcribed [{info.language}]: "{text}"'
            )

            if text and len(text.split()) >= 3:   # ignore 1-2 word noise transcriptions
                msg = String()
                msg.data = text
                self.pub.publish(msg)
            elif text:
                self.get_logger().info(f'Too short to be a command: "{text}" - ignoring. ')
            else:
                self.get_logger().info('Empty transcription — ignoring.')

        except Exception as e:
            self.get_logger().error(f'Transcription failed: {e}')
        finally:
            os.unlink(tmp_path)  # clean up temp file

    def destroy_node(self):
        self.running = False
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


if __name__ == '__main__':
    main()
