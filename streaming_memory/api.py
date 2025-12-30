"""
FastAPI app factory for streaming memory service.

Creates a FastAPI app that can be mounted in any ASGI server.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import AssistantConfig
from .service import StreamingMemoryService


def create_app(
    service: StreamingMemoryService,
    config: AssistantConfig,
    model_id: str = "unknown",
) -> FastAPI:
    """
    Create a FastAPI app for the streaming memory service.

    Args:
        service: The StreamingMemoryService instance
        config: Assistant configuration
        model_id: Model identifier for health endpoint

    Returns:
        FastAPI app ready to be served
    """
    app = FastAPI(title=f"Streaming Memory - {config.name}")

    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "model": model_id,
            "assistant": config.name,
        }

    @app.post("/chat/stream")
    async def chat_stream(request: Request):
        """Stream a chat response with dynamic memory retrieval."""
        data = await request.json()

        message = data.get("message", "")
        history = data.get("history", [])
        update_every_n = data.get("update_every_n", 1)
        max_memories = data.get("max_memories", 5)
        lookback_tokens = data.get("lookback_tokens", 60)

        def generate():
            for event in service.generate_stream(
                message=message,
                history=history,
                update_every_n=update_every_n,
                max_memories=max_memories,
                lookback_tokens=lookback_tokens,
            ):
                yield event.to_sse()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app

