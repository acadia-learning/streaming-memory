"""
WebSocket Voice Agent on Modal - Direct port of LiveKit agent.

Same functionality, simpler transport:
- Client connects via WebSocket (instead of LiveKit room)
- Client sends audio via WebSocket (base64 PCM)
- Server sends transcripts, tokens, TTS back via WebSocket
- Uses DeepGram for STT, vLLM for generation, ElevenLabs for TTS
"""

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-8B"
APP_NAME = "websocket-voice-agent"
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

package_path = Path(__file__).parent.parent / "streaming_memory"

app = modal.App(APP_NAME)

# Image layers
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libavcodec-dev", "libavformat-dev", "libavutil-dev")
    .pip_install("vllm>=0.6", "torch", "transformers>=4.40")
)

deps_image = base_image.pip_install(
    "numpy",
    "huggingface_hub",
    "fastapi[standard]",
    "websockets",
)

def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, ignore_patterns=["*.gguf"])
    print(f"Downloaded {MODEL_ID}")

image = deps_image.run_function(download_model).add_local_dir(package_path, "/root/streaming_memory")

SYSTEM_PROMPT = """You're a friend listening to someone tell you a story. Their speech appears in [TRANSCRIPTION] as they talk.

Keep thinking continuously as new words come in. When you see updated transcription, reconsider: what's new? What does this change? What questions do you have now?

Keep your thinking going - don't stop until they finish speaking. When they finish, close your thinking and respond naturally like a friend. Keep your response concise - just a few sentences.

/think"""


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    scaledown_window=300,
    secrets=[
        modal.Secret.from_name("deepgram-secret"),
        modal.Secret.from_name("elevenlabs-secret"),
    ],
)
@modal.asgi_app()
def web_app():
    """FastAPI app with WebSocket endpoint and loaded vLLM model."""
    from fastapi import FastAPI, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.websockets import WebSocketDisconnect
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import websockets

    # Load model at startup
    print(f"Loading {MODEL_ID} with vLLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enforce_eager=True,
    )
    print(f"vLLM loaded!")

    def build_prompt(transcript: str, user_finished: bool = False) -> str:
        if user_finished:
            user_content = f"""[TRANSCRIPTION - COMPLETE]
{transcript}

User finished speaking. Close your thinking with </think> and respond naturally."""
        else:
            user_content = f"""[TRANSCRIPTION - still talking...]
{transcript}
[...]

Keep thinking continuously about what they're saying. What's new? What's interesting? What questions do you have? Keep your thoughts flowing - they're still talking."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # FastAPI app
    fastapi_app = FastAPI()
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_ID}

    @fastapi_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        session_id = f"session-{int(time.time())}"
        print(f"[{session_id}] Client connected")

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

        async def send(event_type: str, data: dict = {}):
            msg = json.dumps({"type": event_type, **data})
            try:
                await websocket.send_text(msg)
            except:
                pass

        # Token buffer
        token_buffer = asyncio.Queue()

        async def token_streamer():
            while not stop_generation.is_set():
                try:
                    token_data = await asyncio.wait_for(token_buffer.get(), timeout=0.1)
                    await send("token", token_data)
                except asyncio.TimeoutError:
                    continue

        # TTS streaming
        async def stream_tts(text_queue: asyncio.Queue):
            uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream-input?model_id=eleven_turbo_v2_5&output_format=pcm_24000"
            try:
                async with websockets.connect(uri) as ws:
                    await ws.send(json.dumps({
                        "text": " ",
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                        "xi_api_key": elevenlabs_api_key,
                    }))

                    async def send_text():
                        while True:
                            text = await text_queue.get()
                            if text is None:
                                await ws.send(json.dumps({"text": ""}))
                                break
                            await ws.send(json.dumps({"text": text, "try_trigger_generation": True}))

                    async def receive_audio():
                        await send("tts_start", {})
                        async for message in ws:
                            data = json.loads(message)
                            if "audio" in data and data["audio"]:
                                await send("tts_audio", {"audio": data["audio"]})
                        await send("tts_end", {})

                    await asyncio.gather(send_text(), receive_audio())
            except Exception as e:
                print(f"[{session_id}] TTS error: {e}")

        # Generation loop
        async def generate_with_transcript():
            nonlocal generated_text, current_transcript_snapshot, in_thinking

            tts_queue = None
            tts_task = None

            while not stop_generation.is_set():
                if not live_transcript.strip():
                    await asyncio.sleep(0.1)
                    continue

                transcript_changed = live_transcript != current_transcript_snapshot

                if transcript_changed:
                    await send("context_injection", {"old": current_transcript_snapshot, "new": live_transcript})

                    if not in_thinking:
                        in_thinking = True
                        if tts_task and tts_queue:
                            await tts_queue.put(None)
                            tts_task = None
                        await send("restart_thinking", {"reason": "user_resumed_speaking"})
                        await send("thinking_start", {"reason": "new_transcript"})

                    current_transcript_snapshot = live_transcript
                    generated_text = ""

                prompt = build_prompt(live_transcript, user_finished=user_finished_speaking)
                full_prompt = prompt + generated_text if generated_text else prompt

                if not generated_text:
                    if not transcript_changed:
                        await send("generation_start", {"transcript": live_transcript})
                    await send("thinking_start", {})
                    in_thinking = True

                stop_tokens = ["<|im_end|>", "<|endoftext|>"]
                if user_is_speaking or not user_finished_speaking:
                    stop_tokens.append("</think>")

                def do_generate():
                    params = SamplingParams(temperature=0.7, max_tokens=200, stop=stop_tokens)
                    outputs = llm.generate([full_prompt], params, use_tqdm=False)
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
                        await token_buffer.put({"t": char, "is_thinking": True})
                    for char in '</think>':
                        await token_buffer.put({"t": char, "is_thinking": True})

                    in_thinking = False
                    latency_ms = int((time.time() - speech_end_time) * 1000) if speech_end_time > 0 else None
                    await send("thinking_end", {"latency_ms": latency_ms} if latency_ms else {})

                    if response_part.strip():
                        tts_queue = asyncio.Queue()
                        tts_task = asyncio.create_task(stream_tts(tts_queue))
                        await tts_queue.put(response_part)

                    for char in response_part:
                        await token_buffer.put({"t": char, "is_thinking": False})
                else:
                    for char in new_text:
                        await token_buffer.put({"t": char, "is_thinking": in_thinking})
                    if not in_thinking and tts_task and tts_queue and new_text.strip():
                        await tts_queue.put(new_text)

                if ("<|im_end|>" in new_text or "<|endoftext|>" in new_text) and user_finished_speaking:
                    await send("response_complete", {})
                    if tts_task and tts_queue:
                        await tts_queue.put(None)
                        await tts_task
                        tts_task = None
                    generated_text = ""
                    current_transcript_snapshot = ""
                    in_thinking = True

                await asyncio.sleep(0)

        # Audio queue - fill this before connecting to DeepGram
        audio_queue = asyncio.Queue()

        async def receive_from_client():
            """Receive audio from client and queue it."""
            try:
                while not stop_generation.is_set():
                    try:
                        msg = await websocket.receive_text()
                        data = json.loads(msg)
                        if data.get("type") == "audio":
                            await audio_queue.put(data["audio"])
                        elif data.get("type") == "stop":
                            break
                    except WebSocketDisconnect:
                        print(f"[{session_id}] Client disconnected")
                        break
            except Exception as e:
                print(f"[{session_id}] Client receive error: {e}")
            finally:
                stop_generation.set()

        async def run_deepgram():
            """Connect to DeepGram after first audio, then stream continuously."""
            nonlocal live_transcript, user_is_speaking, user_finished_speaking, speech_end_time, last_speech_time

            SILENCE_THRESHOLD = 1.5

            # Wait for first audio packet
            print(f"[{session_id}] Waiting for first audio...")
            first_audio = await audio_queue.get()
            print(f"[{session_id}] Got first audio, connecting to DeepGram...")

            deepgram_url = "wss://api.deepgram.com/v1/listen?model=nova-2&encoding=linear16&sample_rate=16000&channels=1&interim_results=true&endpointing=300"

            try:
                async with websockets.connect(
                    deepgram_url,
                    additional_headers={"Authorization": f"Token {deepgram_api_key}"},
                    ping_interval=5,
                    ping_timeout=20,
                ) as dg_ws:
                    print(f"[{session_id}] DeepGram connected")

                    # Send first audio immediately
                    audio_bytes = base64.b64decode(first_audio)
                    await dg_ws.send(audio_bytes)
                    print(f"[{session_id}] Sent first audio chunk ({len(audio_bytes)} bytes)")

                    async def forward_to_deepgram():
                        """Forward queued audio to DeepGram."""
                        chunk_count = 1
                        try:
                            while not stop_generation.is_set():
                                try:
                                    audio_b64 = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                                    audio_bytes = base64.b64decode(audio_b64)
                                    await dg_ws.send(audio_bytes)
                                    chunk_count += 1
                                    if chunk_count % 50 == 0:
                                        print(f"[{session_id}] Sent {chunk_count} audio chunks to DeepGram")
                                except asyncio.TimeoutError:
                                    continue
                        except Exception as e:
                            print(f"[{session_id}] DeepGram forward error: {e}")

                    async def process_transcripts():
                        nonlocal live_transcript, user_is_speaking, user_finished_speaking, speech_end_time, last_speech_time

                        try:
                            async for msg in dg_ws:
                                if stop_generation.is_set():
                                    break

                                data = json.loads(msg)
                                msg_type = data.get("type", "")
                                
                                if msg_type == "Metadata":
                                    print(f"[{session_id}] DeepGram metadata received")
                                    continue
                                
                                if msg_type != "Results":
                                    continue

                                channel = data.get("channel", {})
                                alternatives = channel.get("alternatives", [])
                                if not alternatives:
                                    continue

                                transcript_text = alternatives[0].get("transcript", "")
                                is_final = data.get("is_final", False)

                                if not transcript_text:
                                    continue
                                
                                print(f"[{session_id}] Transcript: '{transcript_text}' (final={is_final})")

                                if is_final:
                                    live_transcript = (live_transcript + " " + transcript_text).strip()
                                    user_is_speaking = False
                                    last_speech_time = time.time()

                                    await send("transcript", {
                                        "text": transcript_text,
                                        "is_final": True,
                                        "full_transcript": live_transcript,
                                        "user_speaking": False,
                                    })

                                    async def check_finished():
                                        nonlocal user_finished_speaking, speech_end_time
                                        await asyncio.sleep(SILENCE_THRESHOLD)
                                        if time.time() - last_speech_time >= SILENCE_THRESHOLD and not user_is_speaking:
                                            user_finished_speaking = True
                                            speech_end_time = time.time()
                                            await send("status", {"message": "Generating response...", "stage": "ready_to_respond"})

                                    asyncio.create_task(check_finished())
                                else:
                                    user_is_speaking = True
                                    user_finished_speaking = False
                                    last_speech_time = time.time()

                                    await send("transcript", {
                                        "text": transcript_text,
                                        "is_final": False,
                                        "full_transcript": live_transcript + " " + transcript_text,
                                        "user_speaking": True,
                                    })
                        except Exception as e:
                            print(f"[{session_id}] Transcript error: {e}")

                    await asyncio.gather(
                        forward_to_deepgram(),
                        process_transcripts()
                    )

            except Exception as e:
                print(f"[{session_id}] DeepGram error: {e}")

        # Run session
        streamer_task = asyncio.create_task(token_streamer())
        generation_task = asyncio.create_task(generate_with_transcript())

        await send("status", {"message": "Container started", "stage": "container"})
        await send("status", {"message": "Model ready", "stage": "model"})
        await send("connected", {"session": session_id})
        await send("status", {"message": "Ready! Start speaking.", "stage": "ready"})

        try:
            # Run client receiver and DeepGram handler in parallel
            await asyncio.gather(
                receive_from_client(),
                run_deepgram(),
            )
        except WebSocketDisconnect:
            print(f"[{session_id}] Client disconnected")
        except Exception as e:
            print(f"[{session_id}] Session error: {e}")
        finally:
            stop_generation.set()
            generation_task.cancel()
            streamer_task.cancel()
            print(f"[{session_id}] Session ended")

    return fastapi_app
