"""
Debug and observability utilities for WebSocket voice agent.

Provides structured logging, metrics collection, and real-time monitoring.
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from collections import deque


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: float
    level: LogLevel
    component: str
    message: str
    data: dict[str, Any]
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "component": self.component,
            "message": self.message,
            "data": self.data,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class MetricsSnapshot:
    """A snapshot of performance metrics."""
    timestamp: float
    session_id: str
    
    # Transcription metrics
    transcript_received_at: Optional[float] = None
    transcript_latency_ms: Optional[int] = None
    
    # Thinking metrics
    thinking_started_at: Optional[float] = None
    thinking_ended_at: Optional[float] = None
    thinking_duration_ms: Optional[int] = None
    thinking_tokens: int = 0
    
    # Response metrics
    response_started_at: Optional[float] = None
    response_ended_at: Optional[float] = None
    response_duration_ms: Optional[int] = None
    response_tokens: int = 0
    
    # TTS metrics
    tts_started_at: Optional[float] = None
    tts_ended_at: Optional[float] = None
    tts_duration_ms: Optional[int] = None
    
    # Client metrics
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class SessionMetrics:
    """Track metrics for a single session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        
        # Events
        self.transcript_received_at = None
        self.transcript_latency_ms = None
        
        self.thinking_started_at = None
        self.thinking_ended_at = None
        self.thinking_tokens = 0
        
        self.response_started_at = None
        self.response_ended_at = None
        self.response_tokens = 0
        
        self.tts_started_at = None
        self.tts_ended_at = None
        
        # Bytes/messages
        self.bytes_sent = 0
        self.bytes_received = 0
        self.messages_sent = 0
        self.messages_received = 0
    
    def record_transcript_received(self):
        """Record when transcription is received."""
        self.transcript_received_at = time.time()
        self.transcript_latency_ms = int((self.transcript_received_at - self.start_time) * 1000)
    
    def record_thinking_start(self):
        """Record when thinking starts."""
        self.thinking_started_at = time.time()
    
    def record_thinking_end(self):
        """Record when thinking ends."""
        self.thinking_ended_at = time.time()
    
    def record_thinking_token(self):
        """Increment thinking token count."""
        self.thinking_tokens += 1
    
    def record_response_start(self):
        """Record when response starts."""
        self.response_started_at = time.time()
    
    def record_response_end(self):
        """Record when response ends."""
        self.response_ended_at = time.time()
    
    def record_response_token(self):
        """Increment response token count."""
        self.response_tokens += 1
    
    def record_tts_start(self):
        """Record when TTS starts."""
        self.tts_started_at = time.time()
    
    def record_tts_end(self):
        """Record when TTS ends."""
        self.tts_ended_at = time.time()
    
    def record_bytes_sent(self, num_bytes: int):
        """Record bytes sent."""
        self.bytes_sent += num_bytes
    
    def record_bytes_received(self, num_bytes: int):
        """Record bytes received."""
        self.bytes_received += num_bytes
    
    def record_message_sent(self):
        """Record message sent."""
        self.messages_sent += 1
    
    def record_message_received(self):
        """Record message received."""
        self.messages_received += 1
    
    def get_snapshot(self) -> MetricsSnapshot:
        """Get a snapshot of current metrics."""
        thinking_duration_ms = None
        if self.thinking_started_at and self.thinking_ended_at:
            thinking_duration_ms = int((self.thinking_ended_at - self.thinking_started_at) * 1000)
        
        response_duration_ms = None
        if self.response_started_at and self.response_ended_at:
            response_duration_ms = int((self.response_ended_at - self.response_started_at) * 1000)
        
        tts_duration_ms = None
        if self.tts_started_at and self.tts_ended_at:
            tts_duration_ms = int((self.tts_ended_at - self.tts_started_at) * 1000)
        
        return MetricsSnapshot(
            timestamp=time.time(),
            session_id=self.session_id,
            transcript_received_at=self.transcript_received_at,
            transcript_latency_ms=self.transcript_latency_ms,
            thinking_started_at=self.thinking_started_at,
            thinking_ended_at=self.thinking_ended_at,
            thinking_duration_ms=thinking_duration_ms,
            thinking_tokens=self.thinking_tokens,
            response_started_at=self.response_started_at,
            response_ended_at=self.response_ended_at,
            response_duration_ms=response_duration_ms,
            response_tokens=self.response_tokens,
            tts_started_at=self.tts_started_at,
            tts_ended_at=self.tts_ended_at,
            tts_duration_ms=tts_duration_ms,
            bytes_sent=self.bytes_sent,
            bytes_received=self.bytes_received,
            messages_sent=self.messages_sent,
            messages_received=self.messages_received,
        )


class Logger:
    """Structured logger with circular buffer."""
    
    def __init__(self, component: str, buffer_size: int = 1000):
        self.component = component
        self.buffer = deque(maxlen=buffer_size)
    
    def log(
        self,
        level: LogLevel,
        message: str,
        **data
    ) -> LogEntry:
        """Log a message with optional data."""
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            component=self.component,
            message=message,
            data=data,
        )
        self.buffer.append(entry)
        
        # Also print for immediate visibility
        level_str = level.value
        ts = datetime.fromtimestamp(entry.timestamp).isoformat()
        data_str = json.dumps(data) if data else ""
        print(f"[{ts}] {level_str:6} {self.component:15} {message:40} {data_str}")
        
        return entry
    
    def debug(self, message: str, **data) -> LogEntry:
        return self.log(LogLevel.DEBUG, message, **data)
    
    def info(self, message: str, **data) -> LogEntry:
        return self.log(LogLevel.INFO, message, **data)
    
    def warn(self, message: str, **data) -> LogEntry:
        return self.log(LogLevel.WARN, message, **data)
    
    def error(self, message: str, **data) -> LogEntry:
        return self.log(LogLevel.ERROR, message, **data)
    
    def get_recent(self, n: int = 100) -> list[dict]:
        """Get recent log entries."""
        return [entry.to_dict() for entry in list(self.buffer)[-n:]]


class DebugServer:
    """Simple HTTP server for live debugging."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.sessions: dict[str, SessionMetrics] = {}
        self.logs: dict[str, Logger] = {}
    
    def get_session_metrics(self, session_id: str) -> Optional[MetricsSnapshot]:
        """Get metrics for a session."""
        if session_id in self.sessions:
            return self.sessions[session_id].get_snapshot()
        return None
    
    def get_session_logs(self, session_id: str, n: int = 100) -> list[dict]:
        """Get recent logs for a session."""
        if session_id in self.logs:
            return self.logs[session_id].get_recent(n)
        return []
    
    async def setup_fastapi(self, app):
        """Add debug endpoints to FastAPI app."""
        
        @app.get("/debug/sessions")
        async def list_sessions():
            """List all active sessions."""
            return {
                "sessions": list(self.sessions.keys()),
                "count": len(self.sessions),
            }
        
        @app.get("/debug/session/{session_id}/metrics")
        async def get_metrics(session_id: str):
            """Get metrics for a session."""
            metrics = self.get_session_metrics(session_id)
            if metrics:
                return metrics.to_dict()
            return {"error": "Session not found"}
        
        @app.get("/debug/session/{session_id}/logs")
        async def get_logs(session_id: str, n: int = 100):
            """Get logs for a session."""
            logs = self.get_session_logs(session_id, n)
            return {"logs": logs}
        
        @app.get("/debug/health")
        async def health():
            """Health check."""
            return {
                "status": "ok",
                "active_sessions": len(self.sessions),
            }

