#!/usr/bin/env python3
"""
Latency Testing Script for WebSocket Voice Agent

This script measures the end-to-end latency from when speech ends to when
the LLM response begins. It uses text-to-speech to generate test audio
with real speech content that DeepGram can transcribe.

Usage:
    # Test against Modal deployment
    uv run python scripts/test_latency.py --server wss://your-modal-url.modal.run/ws

    # Test with custom phrase
    uv run python scripts/test_latency.py --server wss://... --phrase "Hello, how are you today?"

    # Run multiple trials for statistics
    uv run python scripts/test_latency.py --server wss://... --trials 5
"""

import argparse
import asyncio
import base64
import json
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Try to import TTS library for generating test audio
try:
    import io

    from gtts import gTTS
    from pydub import AudioSegment
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)


@dataclass
class LatencyMetrics:
    """Metrics collected during a latency test."""
    # Timestamps (epoch seconds)
    audio_send_start: float = 0.0
    audio_send_end: float = 0.0
    first_transcript: float = 0.0
    end_of_turn: float = 0.0  # When Deepgram sends EndOfTurn
    thinking_start: float = 0.0
    thinking_end: float = 0.0
    first_response_token: float = 0.0  # First token after </think>
    first_tts_audio: float = 0.0  # First audio from ElevenLabs
    response_complete: float = 0.0

    # Calculated latencies (ms)
    deepgram_latency: float = 0.0  # audio_end -> EndOfTurn
    llm_latency: float = 0.0  # EndOfTurn -> first_response_token
    tts_latency: float = 0.0  # first_response_token -> first_tts_audio
    total_latency: float = 0.0  # audio_end -> first_response

    # Content
    transcript_text: str = ""
    response_text: str = ""
    thinking_text: str = ""

    # Status
    success: bool = False
    error: Optional[str] = None

    def calculate_latencies(self):
        """Calculate derived latency values."""
        # Deepgram latency: audio end -> EndOfTurn
        if self.end_of_turn and self.audio_send_end and self.end_of_turn > self.audio_send_end:
            self.deepgram_latency = (self.end_of_turn - self.audio_send_end) * 1000

        # LLM latency: EndOfTurn -> first response token
        if self.first_response_token and self.end_of_turn and self.first_response_token > self.end_of_turn:
            self.llm_latency = (self.first_response_token - self.end_of_turn) * 1000

        # TTS latency: first response token -> first TTS audio
        if self.first_tts_audio and self.first_response_token and self.first_tts_audio > self.first_response_token:
            self.tts_latency = (self.first_tts_audio - self.first_response_token) * 1000

        # Total latency: audio end -> first response
        if self.first_response_token and self.audio_send_end and self.first_response_token > self.audio_send_end:
            self.total_latency = (self.first_response_token - self.audio_send_end) * 1000


def generate_test_audio(text: str, output_path: str = None, silence_padding_ms: int = 200) -> bytes:
    """Generate test audio using Google TTS.

    Returns PCM16 audio at 16kHz mono, suitable for DeepGram.
    Adds silence padding at the end to help with end-of-speech detection.
    """
    if not HAS_TTS:
        raise RuntimeError(
            "TTS not available. Install with: pip install gtts pydub\n"
            "Also ensure ffmpeg is installed."
        )

    # Generate TTS audio
    tts = gTTS(text=text, lang='en')

    # Save to buffer
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)

    # Convert to WAV with pydub
    audio = AudioSegment.from_mp3(mp3_buffer)

    # Convert to 16kHz mono
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    # Add silence padding at the end to help trigger end-of-speech detection
    if silence_padding_ms > 0:
        silence = AudioSegment.silent(duration=silence_padding_ms, frame_rate=16000)
        silence = silence.set_channels(1).set_sample_width(2)
        audio = audio + silence
        print(f"Added {silence_padding_ms}ms silence padding")

    # Get raw PCM bytes
    pcm_bytes = audio.raw_data

    # Optionally save to file for inspection
    if output_path:
        audio.export(output_path, format="wav")
        print(f"Saved test audio to: {output_path}")

    return pcm_bytes


def load_wav_file(path: str) -> bytes:
    """Load a WAV file and return PCM16 at 16kHz mono."""
    import wave

    with wave.open(path, 'rb') as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

        print(f"Loaded WAV: {channels}ch, {sample_width}B, {frame_rate}Hz")

        # Convert to numpy for resampling if needed
        if sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Convert to mono if stereo
        if channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # Resample if needed
        if frame_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio) * 16000 / frame_rate)
            audio = signal.resample(audio, num_samples).astype(np.int16)

        return audio.tobytes()


class LatencyTester:
    """Test client that measures latency metrics."""

    def __init__(self, server_url: str, timeout: float = 30.0):
        self.server_url = server_url
        self.timeout = timeout
        self.ws = None
        self.metrics = LatencyMetrics()
        self._done = asyncio.Event()
        self._got_response = False
        self._stop_silence = asyncio.Event()  # Stop sending silence on EndOfTurn

    async def connect(self):
        """Connect to WebSocket server."""
        print(f"🔗 Connecting to {self.server_url}...")

        try:
            self.ws = await asyncio.wait_for(
                websockets.connect(self.server_url),
                timeout=120.0  # Long timeout for cold starts
            )
            print("✅ Connected!")
        except asyncio.TimeoutError:
            raise RuntimeError("Connection timed out after 120s")
        except Exception as e:
            raise RuntimeError(f"Connection failed: {e}")

    async def send_audio(self, pcm_bytes: bytes, realtime: bool = True):
        """Send audio to server, then continue sending silence for turn detection.

        Args:
            pcm_bytes: PCM16 audio at 16kHz
            realtime: If True, send at realistic pace
        """
        self.metrics.audio_send_start = time.time()

        chunk_size = 3200  # 100ms of audio at 16kHz (16000 * 2 bytes * 0.1s)
        total_chunks = len(pcm_bytes) // chunk_size

        print(f"🎙️  Sending {len(pcm_bytes)} bytes of audio ({total_chunks} chunks)...")

        # Send actual audio
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i:i + chunk_size]

            # Encode and send
            audio_b64 = base64.b64encode(chunk).decode('utf-8')
            await self.ws.send(json.dumps({
                "type": "audio",
                "audio": audio_b64
            }))

            # Simulate real-time if requested
            if realtime:
                await asyncio.sleep(0.1)  # 100ms per chunk

        self.metrics.audio_send_end = time.time()
        duration_ms = (self.metrics.audio_send_end - self.metrics.audio_send_start) * 1000
        print(f"✅ Speech audio sent in {duration_ms:.0f}ms")

        # Continue sending silence for Flux turn detection
        # Flux needs continuous audio to detect end of turn
        # Use very low level noise instead of perfect zeros (which can cause issues)
        noise = np.random.randint(-10, 10, chunk_size // 2, dtype=np.int16)  # Very quiet noise
        silence_chunk = base64.b64encode(noise.tobytes()).decode('utf-8')
        print("🔇 Sending silence for turn detection...")

        silence_start = time.time()
        # Stop on: response done, EndOfTurn detected, or 10s max
        while not self._done.is_set() and not self._stop_silence.is_set() and (time.time() - silence_start) < 10.0:
            try:
                await self.ws.send(json.dumps({
                    "type": "audio",
                    "audio": silence_chunk
                }))
                await asyncio.sleep(0.1)  # 100ms chunks
            except Exception:
                break

        print("🔇 Stopped sending silence")

    async def receive_messages(self):
        """Receive and process server messages."""
        try:
            async for message in self.ws:
                if self._done.is_set():
                    break

                try:
                    msg = json.loads(message)
                    msg_type = msg.get("type", "")
                    now = time.time()

                    if msg_type == "connected":
                        print(f"📡 Session: {msg.get('session', 'unknown')}")

                    elif msg_type == "status":
                        stage = msg.get("stage", "")
                        message_text = msg.get("message", "")
                        print(f"ℹ️  Status: {message_text} ({stage})")

                    elif msg_type == "transcript":
                        is_final = msg.get("is_final", False)
                        text = msg.get("text", "")

                        if not self.metrics.first_transcript:
                            self.metrics.first_transcript = now

                        if is_final:
                            if not self.metrics.end_of_turn:  # Only set if not already set by EagerEndOfTurn
                                self.metrics.end_of_turn = now
                            self.metrics.transcript_text = msg.get("full_transcript", text)
                            latency = (now - self.metrics.audio_send_end) * 1000
                            print(f"📝 EndOfTurn ({latency:.0f}ms): {self.metrics.transcript_text}")
                            self._stop_silence.set()  # Stop sending silence to avoid TurnResumed
                        else:
                            print(f"📝 Interim: {text}")

                    elif msg_type == "thinking_start":
                        self.metrics.thinking_start = now
                        print("🧠 Thinking started...")

                    elif msg_type == "thinking_end":
                        self.metrics.thinking_end = now
                        latency_reported = msg.get("latency_ms", 0)
                        print(f"🧠 Thinking ended (server reported: {latency_reported}ms)")

                    elif msg_type == "token":
                        is_thinking = msg.get("is_thinking", True)
                        token = msg.get("t", "")

                        if is_thinking:
                            self.metrics.thinking_text += token
                        else:
                            if not self._got_response:
                                self._got_response = True
                                self.metrics.first_response_token = now
                                self.metrics.success = True
                                total_latency = (now - self.metrics.audio_send_end) * 1000
                                print(f"💬 First response token! Total latency: {total_latency:.0f}ms")
                                self._done.set()  # End test on first response token
                            self.metrics.response_text += token

                    elif msg_type == "response_complete":
                        self.metrics.response_complete = now
                        self.metrics.success = True
                        print(f"✅ Response complete: {self.metrics.response_text[:100]}...")
                        self._done.set()

                    elif msg_type == "EndOfTurn":
                        # Explicit EndOfTurn event (from EagerEndOfTurn)
                        if not self.metrics.end_of_turn:
                            self.metrics.end_of_turn = now
                            self.metrics.transcript_text = msg.get("transcript", "")
                            latency = (now - self.metrics.audio_send_end) * 1000
                            print(f"🚀 EagerEndOfTurn ({latency:.0f}ms): {self.metrics.transcript_text}")
                            self._stop_silence.set()  # Stop sending silence to avoid TurnResumed

                    elif msg_type == "restart_thinking":
                        print(f"🔄 Thinking restarted: {msg.get('reason', '')}")
                        self.metrics.thinking_text = ""

                    elif msg_type == "tts_start":
                        print("🔊 TTS started")

                    elif msg_type == "tts_audio":
                        if not self.metrics.first_tts_audio:
                            self.metrics.first_tts_audio = now
                            if self.metrics.first_response_token:
                                tts_latency = (now - self.metrics.first_response_token) * 1000
                                print(f"🔊 First TTS audio ({tts_latency:.0f}ms after response)")

                    elif msg_type == "tts_end":
                        print("🔊 TTS ended")

                    elif msg_type == "error":
                        self.metrics.error = msg.get("error", "Unknown error")
                        print(f"❌ Error: {self.metrics.error}")
                        self._done.set()

                    else:
                        print(f"❓ Unknown message: {msg_type}")

                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON: {message[:100]}")

        except websockets.exceptions.ConnectionClosed:
            print("🔌 Connection closed")
        except Exception as e:
            self.metrics.error = str(e)
            print(f"❌ Receive error: {e}")

    async def run_test(self, audio_bytes: bytes) -> LatencyMetrics:
        """Run a complete latency test."""
        receive_task = None
        try:
            await self.connect()

            # Start receiving in background - keep reference to prevent GC
            receive_task = asyncio.create_task(self.receive_messages())

            # Wait for ready status (server sends multiple status messages during startup)
            print("⏳ Waiting for server ready...")
            await asyncio.sleep(1.0)

            # Send audio
            await self.send_audio(audio_bytes, realtime=True)

            # Wait for response or timeout
            print(f"⏳ Waiting for response (timeout: {self.timeout}s)...")
            try:
                await asyncio.wait_for(self._done.wait(), timeout=self.timeout)
            except asyncio.TimeoutError:
                self.metrics.error = f"Timeout after {self.timeout}s"
                print(f"⏰ {self.metrics.error}")

            # Calculate latencies
            self.metrics.calculate_latencies()

            return self.metrics

        except Exception as e:
            print(f"❌ Test error: {e}")
            self.metrics.error = str(e)
            return self.metrics

        finally:
            if receive_task:
                receive_task.cancel()
            if self.ws:
                try:
                    await self.ws.close()
                except Exception:
                    pass


def print_metrics_summary(metrics: LatencyMetrics):
    """Print a summary of latency metrics."""
    print("\n" + "=" * 60)
    print("LATENCY METRICS SUMMARY")
    print("=" * 60)

    if metrics.error:
        print(f"❌ Error: {metrics.error}")
        return

    print("✅ Test completed successfully\n")

    print("Timing breakdown:")
    print(f"  • Audio end → EndOfTurn:     {metrics.deepgram_latency:>7.0f}ms  (Deepgram)")
    print(f"  • EndOfTurn → Response:      {metrics.llm_latency:>7.0f}ms  (LLM)")
    print(f"  • Response → TTS audio:      {metrics.tts_latency:>7.0f}ms  (ElevenLabs)")
    print("  ────────────────────────────────────")
    print(f"  • TOTAL (audio end → resp):  {metrics.total_latency:>7.0f}ms")

    print("\nContent:")
    print(f"  • Transcript: {metrics.transcript_text[:80]}...")
    print(f"  • Response: {metrics.response_text[:80]}...")


def print_aggregate_stats(all_metrics: list[LatencyMetrics]):
    """Print aggregate statistics across multiple trials."""
    successful = [m for m in all_metrics if m.success]

    if not successful:
        print("\n❌ No successful trials")
        return

    print("\n" + "=" * 60)
    print(f"AGGREGATE STATISTICS ({len(successful)}/{len(all_metrics)} successful)")
    print("=" * 60)

    latencies = [m.total_latency for m in successful]

    print("\nTotal Latency (audio end → first response):")
    print(f"  • Min:    {min(latencies):>7.0f}ms")
    print(f"  • Max:    {max(latencies):>7.0f}ms")
    print(f"  • Mean:   {np.mean(latencies):>7.0f}ms")
    print(f"  • Median: {np.median(latencies):>7.0f}ms")
    print(f"  • Std:    {np.std(latencies):>7.0f}ms")

    # Component breakdown
    dg_lats = [m.deepgram_latency for m in successful if m.deepgram_latency > 0]
    llm_lats = [m.llm_latency for m in successful if m.llm_latency > 0]
    tts_lats = [m.tts_latency for m in successful if m.tts_latency > 0]

    print("\nComponent Latencies (mean):")
    if dg_lats:
        print(f"  • Deepgram (EndOfTurn):  {np.mean(dg_lats):>7.0f}ms")
    if llm_lats:
        print(f"  • LLM (to response):     {np.mean(llm_lats):>7.0f}ms")
    if tts_lats:
        print(f"  • ElevenLabs (to audio): {np.mean(tts_lats):>7.0f}ms")


async def main():
    parser = argparse.ArgumentParser(
        description="Test WebSocket voice agent latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test against Modal deployment
    uv run python scripts/test_latency.py --server wss://your-app.modal.run/ws

    # Use a custom test phrase
    uv run python scripts/test_latency.py --server wss://... --phrase "Tell me about the weather"

    # Use pre-recorded audio file
    uv run python scripts/test_latency.py --server wss://... --audio-file test.wav

    # Run multiple trials
    uv run python scripts/test_latency.py --server wss://... --trials 5
        """
    )

    parser.add_argument(
        "--server",
        required=True,
        help="WebSocket server URL (e.g., wss://your-app.modal.run/ws)"
    )
    parser.add_argument(
        "--phrase",
        default="Hello, I'm testing the latency of this voice agent.",
        help="Test phrase to speak (used with TTS)"
    )
    parser.add_argument(
        "--audio-file",
        help="Path to pre-recorded WAV file (instead of TTS)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of test trials to run"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout per trial in seconds"
    )
    parser.add_argument(
        "--save-audio",
        help="Save generated TTS audio to this path"
    )

    args = parser.parse_args()

    # Get audio
    if args.audio_file:
        print(f"📁 Loading audio from: {args.audio_file}")
        audio_bytes = load_wav_file(args.audio_file)
    else:
        print(f"🎤 Generating TTS audio for: \"{args.phrase}\"")
        if not HAS_TTS:
            print("❌ TTS not available. Install with: pip install gtts pydub")
            print("   Or provide --audio-file with pre-recorded WAV")
            sys.exit(1)
        audio_bytes = generate_test_audio(args.phrase, args.save_audio)

    print(f"📊 Audio size: {len(audio_bytes)} bytes ({len(audio_bytes) / 32000:.1f}s)")

    # Run trials
    all_metrics = []

    for trial in range(args.trials):
        if args.trials > 1:
            print(f"\n{'=' * 60}")
            print(f"TRIAL {trial + 1}/{args.trials}")
            print("=" * 60)

        tester = LatencyTester(args.server, timeout=args.timeout)
        metrics = await tester.run_test(audio_bytes)
        all_metrics.append(metrics)

        if args.trials == 1:
            print_metrics_summary(metrics)

        # Brief pause between trials
        if trial < args.trials - 1:
            await asyncio.sleep(2.0)

    # Aggregate stats for multiple trials
    if args.trials > 1:
        print_aggregate_stats(all_metrics)


if __name__ == "__main__":
    asyncio.run(main())

