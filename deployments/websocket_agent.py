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
import time
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-32B"
APP_NAME = "websocket-voice-agent"
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

package_path = Path(__file__).parent.parent / "streaming_memory"

app = modal.App(APP_NAME)

# Optimized image - combine pip installs into fewer layers
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "vllm>=0.6",
        "torch",
        "transformers>=4.40",
        "numpy",
        "huggingface_hub",
        "fastapi[standard]",
        "websockets",
        "deepgram-sdk",
        extra_options="--no-cache-dir",
    )
    .run_commands(
        f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL_ID}', ignore_patterns=['*.gguf'])\""
    )
    .add_local_dir(package_path, "/root/streaming_memory")
)

SYSTEM_PROMPT = """You're a friend listening to someone tell you a story. Keep responses concise - just a few sentences."""


@app.cls(
    image=image,
    gpu="A100-80GB",
    timeout=600,
    scaledown_window=60,
    secrets=[
        modal.Secret.from_name("deepgram-secret"),
        modal.Secret.from_name("elevenlabs-secret"),
    ],
)
class VoiceAgent:
    @modal.enter()
    def load_model(self):
        """Load model once when container starts - Modal optimizes this."""
        from transformers import AutoTokenizer
        from vllm import LLM

        print(f"[enter] Loading {MODEL_ID} with vLLM...")
        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.llm = LLM(
            model=MODEL_ID,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            enforce_eager=True,
            disable_log_stats=True,
        )

        print(f"[enter] vLLM loaded in {time.time() - start:.1f}s!")

    def build_prompt(self, transcript: str, thinking: str = "", user_finished: bool = False) -> str:
        if user_finished:
            # User finished - include closed thinking, generate response
            user_content = f"{transcript}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            base = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Add the thinking we accumulated (closed) with transition phrase
            if thinking.strip():
                # Add explicit transition to prevent thinking from bleeding into response
                return base + f"<think>{thinking.strip()}\n\nThe user has finished speaking. Now I'll respond:</think>"
            else:
                return base
        else:
            # User still speaking - plan what to say (do NOT respond yet)
            user_content = f"""{transcript} [still speaking...]"""
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT + " Plan what to say while the user is speaking. You will have your chance to speak when the user is done speaking."},
                {"role": "user", "content": user_content},
            ]
            base = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return base + "<think>Planning what to say:"

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app - model is already loaded via @modal.enter()"""
        import websockets
        from fastapi import FastAPI, WebSocket
        from fastapi.middleware.cors import CORSMiddleware
        from starlette.websockets import WebSocketDisconnect
        from vllm import SamplingParams

        # Capture self for use in endpoints
        agent = self

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
            saved_thinking = ""
            current_transcript_snapshot = ""
            in_thinking = True
            user_is_speaking = False
            user_finished_speaking = False
            response_committed = False
            speech_end_time = 0.0
            last_speech_time = 0.0
            stop_generation = asyncio.Event()

            async def send(event_type: str, data: dict = {}):
                msg = json.dumps({"type": event_type, **data})
                try:
                    await websocket.send_text(msg)
                except Exception:
                    pass

            token_buffer = asyncio.Queue()

            async def token_streamer():
                while not stop_generation.is_set():
                    try:
                        token_data = await asyncio.wait_for(token_buffer.get(), timeout=0.1)
                        await send("token", token_data)
                    except asyncio.TimeoutError:
                        continue

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
                                    await ws.send(json.dumps({"text": "", "flush": True}))
                                    break
                                await ws.send(json.dumps({"text": text + " ", "try_trigger_generation": True}))

                        async def receive_audio():
                            await send("tts_start", {})
                            try:
                                async for message in ws:
                                    data = json.loads(message)
                                    if "audio" in data and data["audio"]:
                                        await send("tts_audio", {"audio": data["audio"]})
                                    if data.get("isFinal"):
                                        break
                            except websockets.exceptions.ConnectionClosed:
                                pass
                            await send("tts_end", {})

                        await asyncio.gather(send_text(), receive_audio())
                except Exception:
                    pass

            async def generate_with_transcript():
                nonlocal generated_text, saved_thinking, current_transcript_snapshot, in_thinking, response_committed

                tts_queue = None
                tts_task = None
                forced_end_thinking = False
                empty_generation_count = 0
                thinking_stalled = False

                while not stop_generation.is_set():
                    if not live_transcript.strip():
                        await asyncio.sleep(0.1)
                        continue

                    transcript_changed = live_transcript != current_transcript_snapshot

                    if transcript_changed:
                        await send("context_injection", {"old": current_transcript_snapshot, "new": live_transcript})

                        # If we were stalled, append transcript update to thinking and continue
                        if thinking_stalled and in_thinking and not user_finished_speaking:
                            print(f"[{session_id}] Stall recovery: appending transcript update to thinking")

                            transcript_injection = f"\n\n---\n[USER CONTINUED SPEAKING]\nFull transcript so far: \"{live_transcript}\"\n\nRevising my plan based on new information:"
                            generated_text += transcript_injection

                            for char in transcript_injection:
                                await token_buffer.put({"t": char, "is_thinking": True})

                            await send("thinking_unstalled", {"reason": "transcript_injected", "transcript": live_transcript})
                            thinking_stalled = False
                            empty_generation_count = 0
                            current_transcript_snapshot = live_transcript
                            continue

                        if thinking_stalled:
                            print(f"[{session_id}] Stall cleared by new transcript")
                            await send("thinking_unstalled", {"reason": "new_transcript"})
                        thinking_stalled = False
                        empty_generation_count = 0

                        if not in_thinking and not user_finished_speaking and not response_committed:
                            in_thinking = True
                            generated_text = ""
                            saved_thinking = ""
                            forced_end_thinking = False
                            if tts_task and tts_queue:
                                await tts_queue.put(None)
                                tts_task = None
                            await send("restart_thinking", {"reason": "user_resumed_speaking"})
                            await send("thinking_start", {"reason": "new_transcript"})

                        current_transcript_snapshot = live_transcript

                    if user_finished_speaking and in_thinking and not forced_end_thinking:
                        current_transcript_snapshot = live_transcript
                        saved_thinking = generated_text

                        in_thinking = False
                        forced_end_thinking = True
                        latency_ms = int((time.time() - speech_end_time) * 1000) if speech_end_time > 0 else None
                        await send("thinking_end", {"latency_ms": latency_ms} if latency_ms else {})

                        generated_text = ""

                        if not tts_queue:
                            tts_queue = asyncio.Queue()
                            tts_task = asyncio.create_task(stream_tts(tts_queue))

                    if in_thinking:
                        full_prompt = agent.build_prompt(live_transcript, thinking="", user_finished=False) + generated_text
                    else:
                        full_prompt = agent.build_prompt(live_transcript, thinking=saved_thinking, user_finished=True) + generated_text

                    if transcript_changed or (user_finished_speaking and not response_committed):
                        await send("context", {"prompt": full_prompt})

                    if not generated_text and in_thinking:
                        if not transcript_changed:
                            await send("generation_start", {"transcript": live_transcript})
                        await send("thinking_start", {})

                    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
                    if in_thinking:
                        stop_tokens.append("</think>")

                    def do_generate():
                        params = SamplingParams(temperature=0.7, max_tokens=15, stop=stop_tokens)
                        outputs = agent.llm.generate([full_prompt], params, use_tqdm=False)
                        if outputs and outputs[0].outputs:
                            return outputs[0].outputs[0].text
                        return ""

                    loop = asyncio.get_event_loop()
                    new_text = await loop.run_in_executor(None, do_generate)

                    if stop_generation.is_set():
                        break

                    if not new_text or len(new_text.strip()) == 0:
                        empty_generation_count += 1

                        if in_thinking and not user_finished_speaking and empty_generation_count >= 8 and len(generated_text) > 30:
                            if not thinking_stalled:
                                thinking_stalled = True
                                stall_info = {
                                    "reason": "empty_generations",
                                    "empty_count": empty_generation_count,
                                    "thinking_length": len(generated_text),
                                    "transcript_length": len(live_transcript),
                                }
                                print(f"[{session_id}] THINKING STALLED (empty): {stall_info}")
                                await send("thinking_stalled", stall_info)

                        await asyncio.sleep(0.05)
                        continue

                    empty_generation_count = 0
                    time.time()

                    if in_thinking and not user_finished_speaking:
                        if len(new_text) < 5 and len(generated_text) > 50:
                            pass

                    if in_thinking and not user_finished_speaking and len(generated_text) > 100:
                        recent = generated_text[-200:] if len(generated_text) > 200 else generated_text
                        if new_text in recent[:-len(new_text)] and len(new_text) > 3:
                            if not thinking_stalled:
                                thinking_stalled = True
                                stall_info = {
                                    "reason": "repetition_detected",
                                    "repeated_text": new_text[:50],
                                    "thinking_length": len(generated_text),
                                }
                                print(f"[{session_id}] THINKING STALLED (repetition): {stall_info}")
                                await send("thinking_stalled", stall_info)

                    generated_text += new_text

                    if new_text.strip():
                        for char in new_text:
                            if char.strip() or char in ' \n':
                                await token_buffer.put({"t": char, "is_thinking": in_thinking})

                        if not in_thinking:
                            response_committed = True
                            if tts_task and tts_queue:
                                await tts_queue.put(new_text.strip())

                    if ("<|im_end|>" in new_text or "<|endoftext|>" in new_text) and user_finished_speaking:
                        await send("response_complete", {})
                        if tts_task and tts_queue:
                            await tts_queue.put(None)
                            await tts_task
                            tts_task = None
                        generated_text = ""
                        saved_thinking = ""
                        current_transcript_snapshot = ""
                        in_thinking = True
                        response_committed = False

                    await asyncio.sleep(0)

            audio_queue = asyncio.Queue()

            async def receive_from_client():
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
                            break
                except Exception:
                    pass
                finally:
                    stop_generation.set()

            async def run_deepgram():
                nonlocal live_transcript, user_is_speaking, user_finished_speaking, speech_end_time, last_speech_time

                from deepgram import AsyncDeepgramClient
                from deepgram.core.events import EventType

                try:
                    first_audio = await asyncio.wait_for(audio_queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    return

                if stop_generation.is_set():
                    return

                try:
                    client = AsyncDeepgramClient(api_key=deepgram_api_key)

                    deepgram_conn_ctx = client.listen.v2.connect(
                        model="flux-general-en",
                        encoding="linear16",
                        sample_rate="16000",
                        eot_threshold="0.7",
                        eager_eot_threshold="0.45",
                        eot_timeout_ms="6000",
                    )

                    async with deepgram_conn_ctx as connection:

                        def on_deepgram_message(message):
                            nonlocal live_transcript, user_is_speaking, user_finished_speaking, speech_end_time, last_speech_time
                            nonlocal saved_thinking, generated_text, in_thinking

                            event = getattr(message, "event", None)
                            transcript = getattr(message, "transcript", None)

                            if event == "StartOfTurn":
                                user_is_speaking = True
                                user_finished_speaking = False
                                asyncio.create_task(send("status", {"message": "Listening...", "stage": "listening"}))

                            elif event == "EagerEndOfTurn":
                                user_is_speaking = False
                                user_finished_speaking = True
                                speech_end_time = time.time()

                                if transcript and transcript.strip():
                                    live_transcript = transcript
                                    asyncio.create_task(send("transcript", {
                                        "text": transcript,
                                        "is_final": True,
                                        "full_transcript": transcript,
                                        "user_speaking": False,
                                    }))
                                    asyncio.create_task(send("EndOfTurn", {"transcript": transcript}))
                                    asyncio.create_task(send("status", {"message": "Generating response...", "stage": "ready_to_respond"}))

                            elif event == "TurnResumed":
                                if not response_committed:
                                    user_is_speaking = True
                                    user_finished_speaking = False
                                    if saved_thinking:
                                        generated_text = saved_thinking
                                        in_thinking = True
                                    asyncio.create_task(send("status", {"message": "Listening...", "stage": "listening"}))

                            elif event == "EndOfTurn":
                                if not user_finished_speaking:
                                    user_is_speaking = False
                                    user_finished_speaking = True
                                    speech_end_time = time.time()

                                    if transcript and transcript.strip():
                                        live_transcript = transcript
                                        asyncio.create_task(send("transcript", {
                                            "text": transcript,
                                            "is_final": True,
                                            "full_transcript": transcript,
                                            "user_speaking": False,
                                        }))
                                        asyncio.create_task(send("status", {"message": "Generating response...", "stage": "ready_to_respond"}))

                            elif transcript:
                                live_transcript = transcript
                                last_speech_time = time.time()
                                asyncio.create_task(send("transcript", {
                                    "text": transcript,
                                    "is_final": False,
                                    "full_transcript": transcript,
                                    "user_speaking": True,
                                }))

                        connection.on(EventType.MESSAGE, on_deepgram_message)
                        connection.on(EventType.ERROR, lambda err: print(f"[{session_id}] Deepgram error: {err}"))

                        listen_task = asyncio.create_task(connection.start_listening())

                        audio_bytes = base64.b64decode(first_audio)
                        await connection._send(audio_bytes)

                        try:
                            while not stop_generation.is_set():
                                try:
                                    audio_b64 = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                                    audio_bytes = base64.b64decode(audio_b64)
                                    await connection._send(audio_bytes)
                                except asyncio.TimeoutError:
                                    continue
                        except Exception as e:
                            print(f"[{session_id}] Flux error: {e}")
                        finally:
                            listen_task.cancel()

                except Exception as e:
                    print(f"[{session_id}] DeepGram Flux error: {e}")
                    import traceback
                    traceback.print_exc()

            streamer_task = asyncio.create_task(token_streamer())
            generation_task = asyncio.create_task(generate_with_transcript())

            await send("status", {"message": "Container started", "stage": "container"})
            await send("status", {"message": "Model ready", "stage": "model"})
            await send("connected", {"session": session_id})
            await send("status", {"message": "Ready! Start speaking.", "stage": "ready"})

            try:
                receive_task = asyncio.create_task(receive_from_client())
                deepgram_task = asyncio.create_task(run_deepgram())

                done, pending = await asyncio.wait(
                    [receive_task, deepgram_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"[{session_id}] Error: {e}")
            finally:
                stop_generation.set()
                for task in [generation_task, streamer_task]:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                print(f"[{session_id}] Disconnected")

        return fastapi_app
