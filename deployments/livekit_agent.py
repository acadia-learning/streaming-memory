"""
LiveKit Agent on Modal - Real-time voice with streaming LLM and TTS.

This agent:
1. Joins a LiveKit room
2. Listens for audio via DeepGram STT
3. Injects transcription into LLM context mid-generation
4. Streams reasoning back to the room
5. Speaks responses via ElevenLabs TTS (sent via data channel)

Run:
    modal deploy deployments/livekit_agent.py
"""

import asyncio
import os
import sys
from pathlib import Path

import modal

# Configuration
MODEL_ID = "Qwen/Qwen3-8B"
APP_NAME = "livekit-voice-agent"
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George - natural conversational voice

package_path = Path(__file__).parent.parent / "streaming_memory"


app = modal.App(APP_NAME)

# Base image with heavy deps - this layer is cached and rarely rebuilt
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libavcodec-dev", "libavformat-dev", "libavutil-dev")
    .pip_install(
        # Heavy deps first - cached as long as these don't change
        "vllm>=0.6",
        "torch",
        "transformers>=4.40",
    )
)

# Light deps layer - faster to rebuild
deps_image = base_image.pip_install(
    "numpy",
    "huggingface_hub",
    "fastapi[standard]",
    "aiohttp",
    "av",
    "livekit",
    "livekit-agents>=0.8",
    "livekit-plugins-deepgram>=0.6",
    "livekit-plugins-silero>=0.6",
    "websockets",  # For ElevenLabs streaming
)

# Download model in a separate step - heavily cached
def download_model():
    """Download LLM during image build."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, ignore_patterns=["*.gguf"])
    print(f"Downloaded {MODEL_ID}")

image = deps_image.run_function(download_model).add_local_dir(package_path, "/root/streaming_memory")

SYSTEM_PROMPT = """You're a friend listening to someone tell you a story. Their speech appears in [TRANSCRIPTION] as they talk.

Keep thinking continuously as new words come in. When you see updated transcription, reconsider: what's new? What does this change? What questions do you have now?

Keep your thinking going - don't stop until they finish speaking. When they finish, close your thinking and respond naturally like a friend. Keep your response concise - just a few sentences.

/think"""


@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    scaledown_window=300,
    secrets=[
        modal.Secret.from_name("livekit-secret"),
        modal.Secret.from_name("deepgram-secret"),
        modal.Secret.from_name("elevenlabs-secret"),
    ],
)
class VoiceAgent:
    """LiveKit Voice Agent with streaming LLM via vLLM and TTS via ElevenLabs."""
    
    @modal.enter()
    def startup(self):
        """Load the LLM model with vLLM."""
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        
        sys.path.insert(0, "/root")
        
        print(f"Loading {MODEL_ID} with vLLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.llm = LLM(
            model=MODEL_ID,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            enforce_eager=True,  # Skip torch.compile for fast startup
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1024,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        print(f"vLLM loaded! Ready for fast inference.")
    
    def build_prompt(self, transcript: str, history: list[dict], user_finished: bool = False) -> str:
        """Build prompt with live transcription."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        for h in history[-4:]:
            messages.append(h)
        
        if user_finished:
            user_content = f"""[TRANSCRIPTION - COMPLETE]
{transcript}

User finished speaking. Close your thinking with </think> and respond naturally."""
        else:
            user_content = f"""[TRANSCRIPTION - still talking...]
{transcript}
[...]

Keep thinking continuously about what they're saying. What's new? What's interesting? What questions do you have? Keep your thoughts flowing - they're still talking."""
        
        messages.append({"role": "user", "content": user_content})
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    @modal.method()
    async def run_agent(self, room_name: str):
        """
        Run the voice agent in a LiveKit room.
        """
        import time
        import json
        import base64
        import aiohttp
        import websockets
        from vllm import SamplingParams
        from livekit import rtc, api
        from livekit.plugins import deepgram
        
        livekit_url = os.environ["LIVEKIT_URL"]
        api_key = os.environ["LIVEKIT_API_KEY"]
        api_secret = os.environ["LIVEKIT_API_SECRET"]
        deepgram_api_key = os.environ["DEEPGRAM_API_KEY"]
        elevenlabs_api_key = os.environ["ELEVEN_API_KEY"]
        
        # State
        live_transcript = ""
        generated_text = ""
        current_transcript_snapshot = ""
        in_thinking = True
        user_is_speaking = False
        user_finished_speaking = False
        speech_end_time = 0.0
        last_speech_time = 0.0
        stop_generation = asyncio.Event()
        generation_task = None
        
        # Create our own HTTP session for DeepGram
        http_session = aiohttp.ClientSession()
        
        # Create room token
        token = api.AccessToken(api_key, api_secret)
        token.with_identity("voice-agent")
        token.with_name("Voice Agent")
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
        ))
        
        # Connect to room with options for better connectivity
        room = rtc.Room()
        
        # Connection options - enable TURN relay for better NAT traversal
        connect_options = rtc.RoomOptions(
            auto_subscribe=True,
            dynacast=False,  # Disable for simpler connection
        )
        
        # Data channel for sending text and audio back to client
        async def send_to_client(event_type: str, data: dict):
            """Send event to client via data channel."""
            msg = json.dumps({"type": event_type, **data})
            try:
                await room.local_participant.publish_data(
                    msg.encode(),
                    reliable=True,
                )
            except Exception as e:
                print(f"Failed to send: {e}")
        
        # Token buffer for smooth output
        token_buffer = asyncio.Queue()
        
        async def token_streamer():
            """Stream tokens from buffer immediately - no delay."""
            while not stop_generation.is_set():
                try:
                    token_data = await asyncio.wait_for(token_buffer.get(), timeout=0.1)
                    await send_to_client("token", token_data)
                except asyncio.TimeoutError:
                    continue
        
        # Start token streamer
        streamer_task = asyncio.create_task(token_streamer())
        
        # ElevenLabs TTS streaming - sends audio chunks over data channel
        async def stream_tts_to_client(text_queue: asyncio.Queue):
            """Stream text to ElevenLabs and audio back to client via data channel."""
            uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream-input?model_id=eleven_turbo_v2_5&output_format=pcm_24000"
            
            try:
                async with websockets.connect(uri) as ws:
                    # Send initial config
                    await ws.send(json.dumps({
                        "text": " ",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75,
                        },
                        "xi_api_key": elevenlabs_api_key,
                    }))
                    
                    async def send_text():
                        """Send text chunks to ElevenLabs."""
                        try:
                            while True:
                                text = await text_queue.get()
                                if text is None:
                                    await ws.send(json.dumps({"text": ""}))
                                    break
                                await ws.send(json.dumps({"text": text, "try_trigger_generation": True}))
                        except Exception as e:
                            print(f"TTS send error: {e}")
                    
                    async def receive_audio():
                        """Receive audio from ElevenLabs and send to client."""
                        try:
                            await send_to_client("tts_start", {})
                            async for message in ws:
                                data = json.loads(message)
                                if "audio" in data and data["audio"]:
                                    # Send base64 audio directly to client
                                    await send_to_client("tts_audio", {"audio": data["audio"]})
                            await send_to_client("tts_end", {})
                        except websockets.exceptions.ConnectionClosed:
                            pass
                        except Exception as e:
                            print(f"TTS receive error: {e}")
                    
                    await asyncio.gather(send_text(), receive_audio())
                    
            except Exception as e:
                print(f"TTS stream error: {e}")
        
        async def generate_with_transcript():
            """Generate LLM response using vLLM, continuously thinking as transcript updates."""
            nonlocal generated_text, current_transcript_snapshot, in_thinking
            
            tts_queue = None
            tts_task = None
            
            while not stop_generation.is_set():
                if not live_transcript.strip():
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if transcript changed
                transcript_changed = live_transcript != current_transcript_snapshot
                
                if transcript_changed:
                    await send_to_client("context_injection", {
                        "old": current_transcript_snapshot,
                        "new": live_transcript,
                    })
                    
                    if not in_thinking:
                        in_thinking = True
                        # Stop any ongoing TTS
                        if tts_task and tts_queue:
                            await tts_queue.put(None)
                            tts_task = None
                        await send_to_client("restart_thinking", {"reason": "user_resumed_speaking"})
                        await send_to_client("thinking_start", {"reason": "new_transcript"})
                    
                    current_transcript_snapshot = live_transcript
                    generated_text = ""
                
                prompt = self.build_prompt(live_transcript, [], user_finished=user_finished_speaking)
                
                if generated_text:
                    full_prompt = prompt + generated_text
                else:
                    full_prompt = prompt
                    if not transcript_changed:
                        await send_to_client("generation_start", {"transcript": live_transcript})
                    await send_to_client("thinking_start", {})
                    in_thinking = True
                
                if user_is_speaking or not user_finished_speaking:
                    stop_tokens = ["<|im_end|>", "<|endoftext|>", "</think>"]
                else:
                    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
                
                def do_generate():
                    params = SamplingParams(
                        temperature=0.7,
                        max_tokens=200,
                        stop=stop_tokens,
                    )
                    outputs = self.llm.generate([full_prompt], params, use_tqdm=False)
                    return outputs[0].outputs[0].text if outputs else ""
                
                loop = asyncio.get_event_loop()
                new_text = await loop.run_in_executor(None, do_generate)
                
                if stop_generation.is_set():
                    break
                
                generated_text += new_text
                
                if '</think>' in new_text:
                    parts = new_text.split('</think>')
                    thinking_part = parts[0]
                    response_part = parts[1] if len(parts) > 1 else ""
                    
                    for char in thinking_part:
                        await token_buffer.put({
                            "t": char,
                            "is_thinking": True,
                            "token_count": len(generated_text),
                        })
                    
                    for char in '</think>':
                        await token_buffer.put({
                            "t": char,
                            "is_thinking": True,
                            "token_count": len(generated_text),
                        })
                    
                    in_thinking = False
                    if speech_end_time > 0:
                        latency_ms = int((time.time() - speech_end_time) * 1000)
                        await send_to_client("thinking_end", {"latency_ms": latency_ms})
                    else:
                        await send_to_client("thinking_end", {})
                    
                    # Start TTS via data channel
                    if response_part.strip():
                        tts_queue = asyncio.Queue()
                        tts_task = asyncio.create_task(stream_tts_to_client(tts_queue))
                        await tts_queue.put(response_part)
                    
                    for char in response_part:
                        await token_buffer.put({
                            "t": char,
                            "is_thinking": False,
                            "token_count": len(generated_text),
                        })
                        
                else:
                    for char in new_text:
                        await token_buffer.put({
                            "t": char,
                            "is_thinking": in_thinking,
                            "token_count": len(generated_text),
                        })
                    
                    if not in_thinking and tts_task and tts_queue and new_text.strip():
                        await tts_queue.put(new_text)
                
                if ("<|im_end|>" in new_text or "<|endoftext|>" in new_text) and user_finished_speaking:
                    await send_to_client("response_complete", {})
                    
                    if tts_task and tts_queue:
                        await tts_queue.put(None)
                        await tts_task
                        tts_task = None
                    
                    generated_text = ""
                    current_transcript_snapshot = ""
                    in_thinking = True
                    continue
                
                await asyncio.sleep(0)
        
        # STT setup
        stt = deepgram.STT(
            api_key=deepgram_api_key,
            http_session=http_session,
        )
        
        @room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            print(f"Participant {participant.identity} disconnected")
            stop_generation.set()
        
        @room.on("disconnected")
        def on_room_disconnected():
            print("Room disconnected")
            stop_generation.set()
        
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
            nonlocal live_transcript, generation_task
            
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            
            print(f"Subscribed to audio from {participant.identity}")
            
            async def process_audio():
                nonlocal live_transcript
                
                audio_stream = rtc.AudioStream(track)
                stt_stream = stt.stream()
                
                async def forward_audio():
                    async for event in audio_stream:
                        stt_stream.push_frame(event.frame)
                
                async def process_transcription():
                    nonlocal live_transcript, user_is_speaking, user_finished_speaking, speech_end_time, last_speech_time
                    import time
                    
                    SILENCE_THRESHOLD = 1.5
                    
                    async for event in stt_stream:
                        if event.type == "interim_transcript":
                            user_is_speaking = True
                            user_finished_speaking = False
                            last_speech_time = time.time()
                            await send_to_client("transcript", {
                                "text": event.alternatives[0].text if event.alternatives else "",
                                "is_final": False,
                                "full_transcript": live_transcript + " " + (event.alternatives[0].text if event.alternatives else ""),
                                "user_speaking": True,
                            })
                        elif event.type == "final_transcript":
                            text = event.alternatives[0].text if event.alternatives else ""
                            live_transcript = (live_transcript + " " + text).strip()
                            user_is_speaking = False
                            last_speech_time = time.time()
                            
                            await send_to_client("transcript", {
                                "text": text,
                                "is_final": True,
                                "full_transcript": live_transcript,
                                "user_speaking": False,
                            })
                            
                            async def check_if_finished():
                                nonlocal user_finished_speaking, speech_end_time
                                await asyncio.sleep(SILENCE_THRESHOLD)
                                if time.time() - last_speech_time >= SILENCE_THRESHOLD and not user_is_speaking:
                                    user_finished_speaking = True
                                    speech_end_time = time.time()
                                    await send_to_client("status", {"message": "Generating response...", "stage": "ready_to_respond"})
                            
                            asyncio.create_task(check_if_finished())
                
                await asyncio.gather(forward_audio(), process_transcription())
            
            asyncio.create_task(process_audio())
            
            if generation_task is None:
                generation_task = asyncio.create_task(generate_with_transcript())
        
        try:
            print(f"Connecting to room: {room_name}")
            
            # Connect with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await asyncio.wait_for(
                        room.connect(livekit_url, token.to_jwt(), options=connect_options),
                        timeout=30.0
                    )
                    break
                except asyncio.TimeoutError:
                    print(f"Connection attempt {attempt + 1} timed out")
                    if attempt == max_retries - 1:
                        raise Exception("Failed to connect after retries")
                    await asyncio.sleep(2)
                except Exception as e:
                    print(f"Connection attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2)
            
            print(f"Connected! Room: {room.name}, Local participant: {room.local_participant.identity}")
            
            # Small delay to ensure connection is stable
            await asyncio.sleep(0.5)
            
            # Send status updates (no audio track publishing - just data channel)
            await send_to_client("status", {"message": "Container started", "stage": "container"})
            await send_to_client("status", {"message": "Connected to room", "stage": "connecting"})
            await send_to_client("status", {"message": "Model ready", "stage": "model"})
            await send_to_client("connected", {"room": room_name})
            await send_to_client("status", {"message": "Ready! Start speaking.", "stage": "ready"})
            
            while not stop_generation.is_set():
                await asyncio.sleep(1)
            
            print("Shutting down agent - participant disconnected")
                
        except Exception as e:
            print(f"Error: {e}")
            raise
        finally:
            stop_generation.set()
            if generation_task:
                generation_task.cancel()
            await room.disconnect()
            await http_session.close()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("livekit-secret")],
    timeout=60,
)
@modal.fastapi_endpoint(method="GET")
def join_room(room_name: str = None, participant_name: str = None):
    """
    Create a room token and spawn the agent.
    """
    from livekit import api
    from fastapi.responses import JSONResponse
    
    if not room_name:
        room_name = f"voice-{int(__import__('time').time())}"
    if not participant_name:
        participant_name = f"user-{__import__('random').randint(1000, 9999)}"
    
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    livekit_url = os.environ["LIVEKIT_URL"]
    
    token = api.AccessToken(api_key, api_secret)
    token.with_identity(participant_name)
    token.with_name(participant_name)
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
    ))
    
    agent = VoiceAgent()
    agent.run_agent.spawn(room_name)
    
    return JSONResponse(
        content={
            "token": token.to_jwt(),
            "room": room_name,
            "livekit_url": livekit_url,
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


@app.local_entrypoint()
def main(room_name: str = "test-room"):
    """Test the agent locally."""
    agent = VoiceAgent()
    agent.run_agent.remote(room_name)
