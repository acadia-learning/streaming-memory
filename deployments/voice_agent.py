"""
Voice Agent deployment on Modal with DeepGram real-time transcription.

Run:
    modal deploy deployments/voice_agent.py

This creates a WebSocket endpoint that:
1. Receives audio from the client
2. Streams directly to DeepGram for real-time transcription
3. Shows words as they're spoken (interim results)
4. Injects transcription into LLM context during generation
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import modal

# Configuration
MODEL_ID = "Qwen/Qwen3-4B"  # Smaller model for faster responses
APP_NAME = "voice-agent"

# Paths
package_path = Path(__file__).parent.parent / "streaming_memory"


def download_model():
    """Download LLM during image build."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, ignore_patterns=["*.gguf"])
    print(f"Downloaded {MODEL_ID}")


# Create Modal app and image
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "accelerate>=0.28",
        "numpy",
        "huggingface_hub",
        "fastapi",
        "uvicorn",
        "pydantic",
        "sentencepiece>=0.1.99",
        "websockets>=12.0",
    )
    .run_function(download_model)
    .add_local_dir(package_path, "/root/streaming_memory")
)

# System prompt for streaming transcription demo
SYSTEM_PROMPT = """You are a helpful assistant engaged in a real-time conversation.

The user is speaking to you, and their speech is being transcribed in real-time. You will see their words appear as they speak in the [LIVE TRANSCRIPTION] section below.

Your task:
1. Analyze what the user is saying so far
2. Consider what they might be asking or talking about
3. Think through how you would respond
4. Draft your potential response in your thinking

Use <think>...</think> tags to reason through your response. Be specific about what you would say back to the user. When you're ready, close your thinking and give your response.

Keep your final response concise and conversational."""


@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    secrets=[modal.Secret.from_name("deepgram-secret")],
)
class VoiceAgent:
    """Voice agent with real-time DeepGram transcription and LLM reasoning."""

    @modal.enter()
    def startup(self):
        """Initialize LLM model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        sys.path.insert(0, "/root")

        # Load LLM
        print(f"Loading {MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"LLM loaded! Device: {next(self.model.parameters()).device}")
        print("Voice agent ready!")

    def build_prompt(self, live_transcript: str, history: list[dict]) -> str:
        """Build the prompt with live transcription injected."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Add conversation history
        for h in history[-4:]:
            messages.append(h)

        # Add live transcription as current user input
        user_content = f"""[LIVE TRANSCRIPTION - User is speaking...]
{live_transcript}
[...]

Based on what has been said so far, respond thoughtfully."""

        messages.append({"role": "user", "content": user_content})

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @modal.asgi_app()
    def serve(self):
        """Create the FastAPI app with WebSocket endpoint."""
        import torch
        import websockets
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware

        api = FastAPI(title="Voice Agent - Streaming Transcription Demo")

        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @api.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_ID}

        @api.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """
            WebSocket endpoint for streaming transcription + LLM reasoning.

            Sends all audio directly to DeepGram for real-time transcription.
            Words appear as soon as they're recognized (interim results).
            """
            await websocket.accept()

            deepgram_ws = None
            deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")

            # State
            conversation_history = []
            live_transcript = ""
            interim_transcript = ""  # Current interim (non-final) words
            generation_task = None
            stop_generation = asyncio.Event()

            async def send_event(event_type: str, data: dict = None):
                """Send an event to the client."""
                try:
                    event = {"type": event_type, **(data or {})}
                    await websocket.send_json(event)
                except Exception:
                    pass

            async def connect_deepgram():
                """Connect to DeepGram Nova-2 for real-time transcription."""
                nonlocal deepgram_ws

                await send_event("status", {"message": "Connecting to DeepGram...", "stage": "deepgram_connecting"})

                # Optimized for fastest possible transcription
                url = (
                    "wss://api.deepgram.com/v1/listen?"
                    "model=nova-2&"  # nova-2 is faster than nova-3
                    "encoding=linear16&"
                    "sample_rate=16000&"
                    "channels=1&"
                    "interim_results=true&"
                    "endpointing=100&"  # Very short - 100ms silence = end of utterance
                    "vad_events=true"  # Get VAD events for faster speech detection
                )

                print("Connecting to DeepGram Nova-2...")

                deepgram_ws = await websockets.connect(
                    url,
                    additional_headers={"Authorization": f"Token {deepgram_api_key}"},
                    ping_interval=20,
                    ping_timeout=10,
                )
                print("DeepGram connected!")
                await send_event("status", {"message": "Ready - speak now!", "stage": "ready"})
                return deepgram_ws

            async def handle_deepgram_events():
                """Process transcription events from DeepGram."""
                nonlocal live_transcript, interim_transcript

                try:
                    async for message in deepgram_ws:
                        try:
                            data = json.loads(message)
                            event_type = data.get("type", "unknown")

                            if event_type == "Results":
                                channel = data.get("channel", {})
                                alternatives = channel.get("alternatives", [{}])
                                transcript = alternatives[0].get("transcript", "")
                                is_final = data.get("is_final", False)
                                speech_final = data.get("speech_final", False)

                                if transcript:
                                    if is_final:
                                        # Final result - append to live_transcript
                                        live_transcript = (live_transcript + " " + transcript).strip()
                                        interim_transcript = ""
                                        full = live_transcript
                                    else:
                                        # Interim result - update interim_transcript
                                        interim_transcript = transcript
                                        full = (live_transcript + " " + interim_transcript).strip()

                                    # Send every word update immediately
                                    await send_event("transcript", {
                                        "text": transcript,
                                        "is_final": is_final,
                                        "speech_final": speech_final,
                                        "full_transcript": full,
                                    })

                            elif event_type == "SpeechStarted":
                                await send_event("speech_started", {})

                            elif event_type == "UtteranceEnd":
                                await send_event("utterance_end", {})

                        except json.JSONDecodeError:
                            continue

                except websockets.exceptions.ConnectionClosed:
                    await send_event("status", {"message": "DeepGram disconnected", "stage": "disconnected"})

            async def generate_with_transcript():
                """
                Generate with transcript injection mid-generation.

                - Model can close </think> naturally when ready to respond
                - When new transcription arrives, we inject it and re-open <think>
                - This lets the model reason → respond → reason again as user speaks
                """
                all_generated_tokens: list[int] = []
                current_transcript_snapshot = ""
                in_thinking = True  # Track thinking state
                hit_eos = False

                while not stop_generation.is_set():
                    # Wait for some transcript to work with
                    full = (live_transcript + " " + interim_transcript).strip()
                    if not full:
                        await asyncio.sleep(0.1)
                        continue

                    # First time starting generation
                    if not all_generated_tokens:
                        current_transcript_snapshot = full
                        await send_event("generation_start", {"transcript": full})

                        prompt = self.build_prompt(full, conversation_history)
                        current_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

                        await send_event("thinking_start", {})
                        in_thinking = True

                    # If we hit EOS, wait for new transcript before continuing
                    if hit_eos:
                        new_full = (live_transcript + " " + interim_transcript).strip()
                        if new_full == current_transcript_snapshot:
                            await asyncio.sleep(0.1)
                            continue
                        else:
                            # New transcript! Reset and continue
                            hit_eos = False

                    with torch.no_grad():
                        chunk_size = 3

                        outputs = self.model.generate(
                            current_ids,
                            max_new_tokens=chunk_size,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )

                        new_token_ids = outputs[0, current_ids.shape[1]:].tolist()

                        for tid in new_token_ids:
                            token_text = self.tokenizer.decode([tid], skip_special_tokens=False)

                            # Handle thinking tags naturally
                            if '<think>' in token_text:
                                in_thinking = True
                                all_generated_tokens.append(tid)
                                continue

                            if '</think>' in token_text:
                                in_thinking = False
                                all_generated_tokens.append(tid)
                                await send_event("thinking_end", {})
                                continue

                            if tid == self.tokenizer.eos_token_id:
                                hit_eos = True
                                await send_event("response_complete", {})
                                continue

                            all_generated_tokens.append(tid)

                            clean_token = self.tokenizer.decode([tid], skip_special_tokens=True)
                            if clean_token:
                                await send_event("token", {
                                    "t": clean_token,
                                    "is_thinking": in_thinking,
                                    "token_count": len(all_generated_tokens),
                                })

                        # Check if transcript changed
                        new_full = (live_transcript + " " + interim_transcript).strip()

                        if new_full != current_transcript_snapshot and new_full:
                            # Transcript updated! Inject new context
                            await send_event("context_injection", {
                                "old_transcript": current_transcript_snapshot,
                                "new_transcript": new_full,
                                "tokens_preserved": len(all_generated_tokens),
                                "was_thinking": in_thinking,
                            })

                            current_transcript_snapshot = new_full

                            # Rebuild prefix with new transcription
                            new_prompt = self.build_prompt(new_full, conversation_history)
                            new_prefix_ids = self.tokenizer.encode(
                                new_prompt, return_tensors="pt"
                            ).to(self.model.device)

                            # If model had finished thinking, re-open thinking block
                            if not in_thinking:
                                # Add <think> to re-enter thinking mode
                                think_token = self.tokenizer.encode("<think>", add_special_tokens=False)
                                all_generated_tokens.extend(think_token)
                                in_thinking = True
                                await send_event("thinking_start", {"reason": "new_transcript"})

                            # Concatenate: new prefix + all generated tokens
                            if all_generated_tokens:
                                generated_tensor = torch.tensor(
                                    [all_generated_tokens], device=self.model.device
                                )
                                current_ids = torch.cat([new_prefix_ids, generated_tensor], dim=1)
                            else:
                                current_ids = new_prefix_ids

                            hit_eos = False  # Reset EOS flag on new transcript
                        else:
                            current_ids = outputs

                        if stop_generation.is_set():
                            break

                        await asyncio.sleep(0)

            deepgram_task = None

            try:
                await send_event("connected")

                # Connect to DeepGram immediately
                await connect_deepgram()
                deepgram_task = asyncio.create_task(handle_deepgram_events())

                # Start generation task
                generation_task = asyncio.create_task(generate_with_transcript())

                # Handle client messages (audio data)
                while True:
                    message = await websocket.receive()

                    if message["type"] == "websocket.disconnect":
                        break

                    if "bytes" in message:
                        # Send audio directly to DeepGram - no VAD, fastest path
                        audio_bytes = message["bytes"]
                        # Ensure we have actual bytes
                        if audio_bytes and isinstance(audio_bytes, (bytes, bytearray)):
                            if deepgram_ws:
                                try:
                                    await deepgram_ws.send(bytes(audio_bytes))
                                except websockets.exceptions.ConnectionClosed:
                                    print("DeepGram connection closed")
                                    deepgram_ws = None

                    elif "text" in message:
                        try:
                            data = json.loads(message["text"])
                            msg_type = data.get("type")

                            if msg_type == "stop":
                                break
                            elif msg_type == "pause":
                                # Stop generation when mic is turned off
                                stop_generation.set()
                                if generation_task:
                                    generation_task.cancel()
                                    generation_task = None
                                await send_event("status", {"message": "Paused", "stage": "paused"})
                                print("Generation paused by user")
                            elif msg_type == "resume":
                                # Resume generation when mic is turned back on
                                stop_generation.clear()
                                generation_task = asyncio.create_task(generate_with_transcript())
                                await send_event("status", {"message": "Resumed", "stage": "ready"})
                                print("Generation resumed")
                            elif msg_type == "clear":
                                live_transcript = ""
                                interim_transcript = ""
                                conversation_history.clear()
                                await send_event("cleared")
                        except json.JSONDecodeError:
                            pass

            except WebSocketDisconnect:
                print("Client disconnected")
            finally:
                print("Cleaning up session...")
                stop_generation.set()
                if deepgram_ws:
                    try:
                        await deepgram_ws.close()
                        print("DeepGram connection closed")
                    except Exception:
                        pass
                if generation_task:
                    generation_task.cancel()
                    print("Generation task cancelled")
                if deepgram_task:
                    deepgram_task.cancel()
                    print("DeepGram task cancelled")
                print("Session cleanup complete - container will scale down after idle timeout")

        return api

