"""
WebSocket protocol for voice agent communication.

Clean, typed message protocol for client-server voice streaming.
All messages are JSON-serializable for easy debugging and logging.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class MessageType(Enum):
    """Server-to-client message types."""
    # Connection
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

    # Transcription
    TRANSCRIPT_INTERIM = "transcript_interim"
    TRANSCRIPT_FINAL = "transcript_final"

    # Agent state
    THINKING_START = "thinking_start"
    THINKING_END = "thinking_end"
    RESPONSE_START = "response_start"
    RESPONSE_END = "response_end"

    # Streaming content
    THINKING_TEXT = "thinking_text"
    RESPONSE_TEXT = "response_text"
    TTS_START = "tts_start"
    TTS_CHUNK = "tts_chunk"
    TTS_END = "tts_end"

    # Status/observability
    STATUS = "status"
    METRICS = "metrics"


class ClientMessageType(Enum):
    """Client-to-server message types."""
    AUDIO_CHUNK = "audio_chunk"
    END_OF_TURN = "end_of_turn"
    INTERRUPT = "interrupt"


@dataclass
class ServerMessage:
    """Base server message."""
    type: MessageType
    data: dict[str, Any]
    timestamp_ms: Optional[int] = None

    def to_json(self) -> str:
        """Serialize to JSON."""
        payload = {
            "type": self.type.value,
            "data": self.data,
        }
        if self.timestamp_ms is not None:
            payload["timestamp_ms"] = self.timestamp_ms
        return json.dumps(payload)

    @staticmethod
    def from_json(json_str: str) -> "ServerMessage":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return ServerMessage(
            type=MessageType(data["type"]),
            data=data.get("data", {}),
            timestamp_ms=data.get("timestamp_ms"),
        )


@dataclass
class ClientMessage:
    """Base client message."""
    type: ClientMessageType
    data: dict[str, Any]

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
        })

    @staticmethod
    def from_json(json_str: str) -> "ClientMessage":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return ClientMessage(
            type=ClientMessageType(data["type"]),
            data=data.get("data", {}),
        )


# Convenience factories
def msg_connected(room_id: str) -> ServerMessage:
    """Client successfully connected."""
    return ServerMessage(MessageType.CONNECTED, {"room_id": room_id})


def msg_error(error: str, code: str = "UNKNOWN") -> ServerMessage:
    """Error occurred."""
    return ServerMessage(MessageType.ERROR, {"error": error, "code": code})


def msg_transcript_interim(text: str, full_transcript: str) -> ServerMessage:
    """Interim transcription (user still speaking)."""
    return ServerMessage(
        MessageType.TRANSCRIPT_INTERIM,
        {"text": text, "full_transcript": full_transcript}
    )


def msg_transcript_final(text: str, full_transcript: str) -> ServerMessage:
    """Final transcription (user finished speaking)."""
    return ServerMessage(
        MessageType.TRANSCRIPT_FINAL,
        {"text": text, "full_transcript": full_transcript}
    )


def msg_thinking_start(reason: str = "") -> ServerMessage:
    """Agent started thinking."""
    return ServerMessage(MessageType.THINKING_START, {"reason": reason})


def msg_thinking_text(text: str) -> ServerMessage:
    """Thinking stream token."""
    return ServerMessage(MessageType.THINKING_TEXT, {"t": text})


def msg_thinking_end(latency_ms: Optional[int] = None) -> ServerMessage:
    """Agent finished thinking."""
    data = {}
    if latency_ms is not None:
        data["latency_ms"] = latency_ms
    return ServerMessage(MessageType.THINKING_END, data)


def msg_response_start() -> ServerMessage:
    """Agent started responding."""
    return ServerMessage(MessageType.RESPONSE_START, {})


def msg_response_text(text: str) -> ServerMessage:
    """Response stream token."""
    return ServerMessage(MessageType.RESPONSE_TEXT, {"t": text})


def msg_response_end(full_response: str) -> ServerMessage:
    """Agent finished responding."""
    return ServerMessage(MessageType.RESPONSE_END, {"full_response": full_response})


def msg_tts_start() -> ServerMessage:
    """TTS started."""
    return ServerMessage(MessageType.TTS_START, {})


def msg_tts_chunk(audio_base64: str) -> ServerMessage:
    """TTS audio chunk (base64 encoded PCM)."""
    return ServerMessage(MessageType.TTS_CHUNK, {"audio": audio_base64})


def msg_tts_end() -> ServerMessage:
    """TTS finished."""
    return ServerMessage(MessageType.TTS_END, {})


def msg_status(stage: str, message: str = "") -> ServerMessage:
    """Status update."""
    return ServerMessage(
        MessageType.STATUS,
        {"stage": stage, "message": message}
    )


def msg_metrics(
    transcript_latency_ms: Optional[int] = None,
    thinking_latency_ms: Optional[int] = None,
    response_latency_ms: Optional[int] = None,
    **kwargs
) -> ServerMessage:
    """Performance metrics."""
    data = {}
    if transcript_latency_ms is not None:
        data["transcript_latency_ms"] = transcript_latency_ms
    if thinking_latency_ms is not None:
        data["thinking_latency_ms"] = thinking_latency_ms
    if response_latency_ms is not None:
        data["response_latency_ms"] = response_latency_ms
    data.update(kwargs)
    return ServerMessage(MessageType.METRICS, data)

