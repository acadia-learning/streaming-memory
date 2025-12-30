#!/usr/bin/env python3
"""
Test script for WebSocket voice agent.

Tests the agent without Modal - useful for local development and debugging.

Usage:
    uv run python examples/test_websocket_agent.py [--record-audio-file path/to/audio.wav]
"""

import asyncio
import base64
import json
import sys
import time
import warnings
from pathlib import Path

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*invalid escape sequence.*')

import numpy as np
import websockets

# Add streaming_memory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_memory.protocol import ClientMessageType


class TestWebSocketClient:
    """Test client for WebSocket voice agent."""
    
    def __init__(self, server_url: str = "ws://localhost:8000", room_id: str = "test-room"):
        self.server_url = server_url
        self.room_id = room_id
        self.ws = None
        
        # Metrics
        self.start_time = time.time()
        self.messages_received = 0
        self.thinking_tokens = 0
        self.response_tokens = 0
        self.last_message_type = None
    
    async def connect(self):
        """Connect to WebSocket server."""
        url = f"{self.server_url}/ws/voice/{self.room_id}"
        print(f"🔗 Connecting to {url}...")
        
        try:
            self.ws = await websockets.connect(url)
            print(f"✅ Connected!")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            raise
    
    async def send_audio_file(self, audio_file: str):
        """Send audio from a WAV file."""
        import wave
        
        print(f"🎙️  Reading audio from {audio_file}")
        
        with wave.open(audio_file, 'rb') as wav:
            # Verify format
            channels, sample_width, frame_rate, num_frames, _, _ = wav.getparams()
            print(f"   Format: {channels}ch, {sample_width}B, {frame_rate}Hz, {num_frames} frames")
            
            # Convert to 16-bit PCM
            audio_data = wav.readframes(num_frames)
            
            # Split into chunks and send
            chunk_size = 4096
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                msg = json.dumps({
                    "type": "audio_chunk",
                    "data": {"audio": audio_base64}
                })
                
                await self.ws.send(msg)
                
                # Simulate real-time sending with small delay
                await asyncio.sleep(chunk_size / frame_rate / 1000)
                
                if (i // chunk_size) % 10 == 0:
                    print(f"   Sent {i // chunk_size} chunks...")
        
        # Signal end of turn
        print("✋ Signaling end of turn")
        await self.ws.send(json.dumps({
            "type": "end_of_turn",
            "data": {}
        }))
    
    async def send_test_audio(self, duration_sec: float = 3.0):
        """Generate and send test audio (silence for testing)."""
        sample_rate = 16000
        duration_frames = int(duration_sec * sample_rate)
        
        print(f"🎵 Generating {duration_sec}s of test audio at {sample_rate}Hz")
        
        # Generate silence (zeros) for testing
        audio = np.zeros(duration_frames)
        
        # Convert to 16-bit PCM
        pcm16 = (audio * 32767).astype(np.int16)
        
        # Split into chunks
        chunk_size = 4096
        for i in range(0, len(pcm16), chunk_size):
            chunk = pcm16[i:i+chunk_size].tobytes()
            
            audio_base64 = base64.b64encode(chunk).decode('utf-8')
            
            msg = json.dumps({
                "type": "audio_chunk",
                "data": {"audio": audio_base64}
            })
            
            await self.ws.send(msg)
            
            # Simulate real-time sending
            await asyncio.sleep(chunk_size / sample_rate)
            
            if (i // chunk_size) % 5 == 0:
                print(f"   Sent {i // chunk_size} chunks...")
        
        # Signal end of turn
        print("✋ Signaling end of turn")
        await self.ws.send(json.dumps({
            "type": "end_of_turn",
            "data": {}
        }))
    
    async def receive_messages(self):
        """Receive and display messages from server."""
        try:
            async for message in self.ws:
                self.messages_received += 1
                
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    self.last_message_type = msg_type
                    
                    elapsed = time.time() - self.start_time
                    
                    if msg_type == "connected":
                        print(f"✅ [{elapsed:.2f}s] Connected to {data['data']['room_id']}")
                    
                    elif msg_type == "status":
                        print(f"ℹ️  [{elapsed:.2f}s] {data['data'].get('message')}")
                    
                    elif msg_type == "transcript_interim":
                        text = data['data'].get('text', '')
                        print(f"🎤 [{elapsed:.2f}s] (interim) {text}")
                    
                    elif msg_type == "transcript_final":
                        text = data['data'].get('text', '')
                        print(f"✓  [{elapsed:.2f}s] (final) {text}")
                    
                    elif msg_type == "thinking_start":
                        reason = data['data'].get('reason', '')
                        print(f"🧠 [{elapsed:.2f}s] Thinking... ({reason})")
                        self.thinking_tokens = 0
                    
                    elif msg_type == "thinking_text":
                        self.thinking_tokens += 1
                        if self.thinking_tokens % 10 == 0:
                            print(f"   Thinking... {self.thinking_tokens} tokens")
                    
                    elif msg_type == "thinking_end":
                        latency = data['data'].get('latency_ms')
                        print(f"✓  [{elapsed:.2f}s] Thinking done ({self.thinking_tokens} tokens, {latency}ms latency)")
                    
                    elif msg_type == "response_start":
                        print(f"💬 [{elapsed:.2f}s] Responding...")
                        self.response_tokens = 0
                    
                    elif msg_type == "response_text":
                        self.response_tokens += 1
                        if self.response_tokens % 10 == 0:
                            print(f"   Response... {self.response_tokens} tokens")
                    
                    elif msg_type == "response_end":
                        full = data['data'].get('full_response', '')
                        print(f"✓  [{elapsed:.2f}s] Response: {full}")
                    
                    elif msg_type == "tts_start":
                        print(f"🔊 [{elapsed:.2f}s] TTS started")
                    
                    elif msg_type == "tts_chunk":
                        # Don't print every chunk, just log
                        pass
                    
                    elif msg_type == "tts_end":
                        print(f"🔊 [{elapsed:.2f}s] TTS complete")
                    
                    elif msg_type == "error":
                        error = data['data'].get('error', 'Unknown error')
                        print(f"❌ [{elapsed:.2f}s] Error: {error}")
                    
                    else:
                        print(f"? [{elapsed:.2f}s] {msg_type}: {data['data']}")
                
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON: {message}")
        
        except websockets.exceptions.ConnectionClosed:
            print(f"\n🔌 Connection closed after {self.messages_received} messages")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
    
    async def close(self):
        """Close connection."""
        if self.ws:
            await self.ws.close()


async def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test WebSocket voice agent"
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:8000",
        help="WebSocket server URL (default: ws://localhost:8000)"
    )
    parser.add_argument(
        "--audio-file",
        help="Path to WAV file to send (instead of generated audio)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration of test audio in seconds (default: 3.0)"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = TestWebSocketClient(server_url=args.server)
    
    try:
        # Connect
        await client.connect()
        
        # Start receiving messages (non-blocking)
        receive_task = asyncio.create_task(client.receive_messages())
        
        # Send audio
        try:
            if args.audio_file:
                await client.send_audio_file(args.audio_file)
            else:
                await client.send_test_audio(args.duration)
        except Exception as e:
            print(f"❌ Failed to send audio: {e}")
        
        # Wait for response
        print("\n⏳ Waiting for response (press Ctrl+C to stop)...")
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.close()
        print(f"\n📊 Test complete")
        print(f"   Messages received: {client.messages_received}")
        print(f"   Thinking tokens: {client.thinking_tokens}")
        print(f"   Response tokens: {client.response_tokens}")
        print(f"   Duration: {time.time() - client.start_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())

