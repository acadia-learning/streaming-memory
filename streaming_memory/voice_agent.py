"""
Voice agent service with DeepGram Flux for real-time speech transcription.

Handles:
- Real-time audio transcription via DeepGram Flux
- Turn-taking and end-of-turn detection
- LLM response generation with streaming
- Barge-in (interruption) support
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import AsyncGenerator, Callable, Optional


class AgentState(Enum):
    """Voice agent state machine states."""
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class Transcript:
    """A transcription result from DeepGram."""
    text: str
    is_final: bool
    confidence: float
    words: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VoiceEvent:
    """An event in the voice agent pipeline."""
    type: str
    data: dict

    def to_dict(self) -> dict:
        return {"type": self.type, **self.data}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class DeepgramFluxClient:
    """
    Client for DeepGram Flux real-time speech transcription.

    Uses the v2 WebSocket API with the flux-general-en model.
    """

    def __init__(
        self,
        api_key: str | None = None,
        sample_rate: int = 16000,
        encoding: str = "linear16",
        eot_threshold: float = 0.7,
        eager_eot_threshold: float | None = 0.5,  # Enable eager end-of-turn
        eot_timeout_ms: int = 5000,
    ):
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY required")

        self.sample_rate = sample_rate
        self.encoding = encoding
        self.eot_threshold = eot_threshold
        self.eager_eot_threshold = eager_eot_threshold
        self.eot_timeout_ms = eot_timeout_ms

        self._websocket = None
        self._connected = False

    def _build_url(self) -> str:
        """Build the WebSocket URL for Flux."""
        params = [
            "model=flux-general-en",
            f"encoding={self.encoding}",
            f"sample_rate={self.sample_rate}",
            f"eot_threshold={self.eot_threshold}",
            f"eot_timeout_ms={self.eot_timeout_ms}",
        ]
        if self.eager_eot_threshold is not None:
            params.append(f"eager_eot_threshold={self.eager_eot_threshold}")

        return f"wss://api.deepgram.com/v2/listen?{'&'.join(params)}"

    async def connect(self):
        """Establish WebSocket connection to DeepGram Flux."""
        import websockets

        url = self._build_url()
        headers = {"Authorization": f"Token {self.api_key}"}

        self._websocket = await websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
        )
        self._connected = True

    async def disconnect(self):
        """Close the WebSocket connection."""
        if self._websocket:
            await self._websocket.close()
            self._connected = False

    async def send_audio(self, audio_chunk: bytes):
        """Send an audio chunk to DeepGram."""
        if self._websocket and self._connected:
            await self._websocket.send(audio_chunk)

    async def receive_events(self) -> AsyncGenerator[VoiceEvent, None]:
        """
        Receive and parse events from DeepGram Flux.

        Event types:
        - transcript: Interim or final transcription
        - end_of_turn: Speaker finished their turn
        - eager_end_of_turn: Early signal that turn may be ending
        - turn_resumed: User continued speaking (cancel draft response)
        - vad: Voice activity detection events
        """
        if not self._websocket:
            return

        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    event_type = data.get("type", "unknown")

                    if event_type == "Results":
                        # Transcription result
                        channel = data.get("channel", {})
                        alternatives = channel.get("alternatives", [{}])
                        transcript = alternatives[0].get("transcript", "")
                        confidence = alternatives[0].get("confidence", 0)
                        words = alternatives[0].get("words", [])
                        is_final = data.get("is_final", False)

                        if transcript:
                            yield VoiceEvent("transcript", {
                                "text": transcript,
                                "is_final": is_final,
                                "confidence": confidence,
                                "words": words,
                            })

                    elif event_type == "EndOfTurn":
                        yield VoiceEvent("end_of_turn", {
                            "confidence": data.get("confidence", 0),
                        })

                    elif event_type == "EagerEndOfTurn":
                        yield VoiceEvent("eager_end_of_turn", {
                            "confidence": data.get("confidence", 0),
                        })

                    elif event_type == "TurnResumed":
                        yield VoiceEvent("turn_resumed", {})

                    elif event_type == "SpeechStarted":
                        yield VoiceEvent("speech_started", {})

                    elif event_type == "Connected":
                        yield VoiceEvent("connected", {
                            "connection_id": data.get("connection_id"),
                        })

                    elif event_type == "Metadata":
                        yield VoiceEvent("metadata", data)

                    else:
                        # Pass through other events
                        yield VoiceEvent(event_type.lower(), data)

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            yield VoiceEvent("error", {"message": str(e)})


class VoiceAgentService:
    """
    Voice agent that combines DeepGram Flux transcription with LLM generation.

    Key features:
    - Real-time speech-to-text with turn detection
    - Streaming LLM response generation
    - Barge-in support (user can interrupt)
    - State machine for managing conversation flow
    """

    def __init__(
        self,
        deepgram_client: DeepgramFluxClient,
        generate_fn: Callable[[str, list[dict]], AsyncGenerator[str, None]],
        system_prompt: str = "You are a helpful voice assistant.",
    ):
        """
        Initialize the voice agent.

        Args:
            deepgram_client: DeepGram Flux client for transcription
            generate_fn: Async generator function that takes (message, history) and yields tokens
            system_prompt: System prompt for the LLM
        """
        self.deepgram = deepgram_client
        self.generate_fn = generate_fn
        self.system_prompt = system_prompt

        self.state = AgentState.IDLE
        self.conversation_history: list[dict] = []
        self.current_transcript = ""
        self.current_response = ""

        # For managing generation interruption
        self._generation_task: Optional[asyncio.Task] = None
        self._should_interrupt = False

    async def start(self):
        """Start the voice agent by connecting to DeepGram."""
        await self.deepgram.connect()
        self.state = AgentState.LISTENING

    async def stop(self):
        """Stop the voice agent."""
        if self._generation_task:
            self._generation_task.cancel()
        await self.deepgram.disconnect()
        self.state = AgentState.IDLE

    async def process_audio(self, audio_chunk: bytes):
        """Process an incoming audio chunk."""
        await self.deepgram.send_audio(audio_chunk)

    async def run(self) -> AsyncGenerator[VoiceEvent, None]:
        """
        Main event loop for the voice agent.

        Yields events for:
        - state_change: Agent state transitions
        - transcript: User speech transcription
        - token: LLM response tokens
        - end_of_turn: User finished speaking
        - interrupted: LLM response was interrupted
        """
        accumulated_transcript = ""

        async for event in self.deepgram.receive_events():
            # Yield raw transcription events
            yield event

            if event.type == "transcript":
                text = event.data.get("text", "")
                is_final = event.data.get("is_final", False)

                if is_final:
                    accumulated_transcript += " " + text
                    accumulated_transcript = accumulated_transcript.strip()
                else:
                    # Interim result - update current transcript
                    self.current_transcript = accumulated_transcript + " " + text

                # If we're currently speaking, this might be a barge-in
                if self.state == AgentState.SPEAKING:
                    yield VoiceEvent("barge_in_detected", {"text": text})

            elif event.type == "speech_started":
                # User started speaking
                if self.state == AgentState.SPEAKING:
                    # Interrupt current response
                    self._should_interrupt = True
                    yield VoiceEvent("interrupted", {
                        "partial_response": self.current_response
                    })

                self.state = AgentState.LISTENING
                yield VoiceEvent("state_change", {"state": self.state.value})

            elif event.type == "eager_end_of_turn":
                # Start preparing response early
                if accumulated_transcript.strip():
                    yield VoiceEvent("preparing_response", {
                        "transcript": accumulated_transcript.strip()
                    })

            elif event.type == "end_of_turn":
                # User finished speaking - generate response
                final_transcript = accumulated_transcript.strip()
                if final_transcript:
                    self.current_transcript = final_transcript
                    accumulated_transcript = ""

                    # Add to history
                    self.conversation_history.append({
                        "role": "user",
                        "content": final_transcript,
                    })

                    yield VoiceEvent("user_turn_complete", {
                        "transcript": final_transcript
                    })

                    # Generate response
                    self.state = AgentState.THINKING
                    yield VoiceEvent("state_change", {"state": self.state.value})

                    self._should_interrupt = False
                    self.current_response = ""

                    async for token in self.generate_fn(
                        final_transcript,
                        self.conversation_history[:-1],  # Exclude current message
                    ):
                        if self._should_interrupt:
                            break

                        if self.state != AgentState.SPEAKING:
                            self.state = AgentState.SPEAKING
                            yield VoiceEvent("state_change", {"state": self.state.value})

                        self.current_response += token
                        yield VoiceEvent("token", {"t": token})

                    if not self._should_interrupt:
                        # Add assistant response to history
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": self.current_response,
                        })

                        yield VoiceEvent("assistant_turn_complete", {
                            "response": self.current_response
                        })

                    self.state = AgentState.LISTENING
                    yield VoiceEvent("state_change", {"state": self.state.value})

            elif event.type == "turn_resumed":
                # User continued speaking - cancel any draft response
                if self.state in (AgentState.THINKING, AgentState.SPEAKING):
                    self._should_interrupt = True
                    yield VoiceEvent("turn_resumed", {
                        "cancelled_response": self.current_response
                    })

    def get_conversation_history(self) -> list[dict]:
        """Get the full conversation history."""
        return self.conversation_history.copy()

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


def create_voice_agent(
    deepgram_api_key: str,
    generate_fn: Callable,
    system_prompt: str = "You are a helpful voice assistant.",
    sample_rate: int = 16000,
    eager_eot_threshold: float | None = 0.5,
) -> VoiceAgentService:
    """
    Factory function to create a voice agent.

    Args:
        deepgram_api_key: DeepGram API key
        generate_fn: Async generator that yields response tokens
        system_prompt: System prompt for the LLM
        sample_rate: Audio sample rate (default 16kHz)
        eager_eot_threshold: Threshold for early end-of-turn detection

    Returns:
        Configured VoiceAgentService
    """
    client = DeepgramFluxClient(
        api_key=deepgram_api_key,
        sample_rate=sample_rate,
        eager_eot_threshold=eager_eot_threshold,
    )

    return VoiceAgentService(
        deepgram_client=client,
        generate_fn=generate_fn,
        system_prompt=system_prompt,
    )

