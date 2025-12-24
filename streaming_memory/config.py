"""
Configuration for streaming memory deployments.

Define system prompts, memory sources, and deployment-specific settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class MemoryConfig:
    """Configuration for a memory pool."""
    
    # Path to JSON file with memories (each item has "content", optional "emotional_intensity", "created_at")
    memory_file: str | Path
    
    # Prefix shown before memories in the prompt
    memory_prefix: str = "[Memories:]"
    
    # MemoryPool parameters
    softmax_temperature: float = 0.15
    diversity_weight: float = 0.5
    association_weight: float = 0.5


@dataclass
class ModelConfig:
    """Configuration for the LLM."""
    
    model_id: str = "Qwen/Qwen3-8B"
    torch_dtype: str = "bfloat16"
    temperature: float = 0.7
    max_tokens: int = 1500


@dataclass
class AssistantConfig:
    """Complete configuration for a streaming memory assistant."""
    
    # Unique name for this assistant
    name: str
    
    # System prompt for the LLM
    system_prompt: str
    
    # Memory configuration
    memory: MemoryConfig
    
    # Model configuration (optional, uses defaults)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Optional description for UI
    description: str = ""


# Pre-built configurations

FAMILY_ASSISTANT = AssistantConfig(
    name="family-assistant",
    description="Personal assistant with access to your memories and notes",
    system_prompt="""You are a helpful personal assistant who has access to the user's memories and notes.

You help them think through decisions by drawing on what you know about their life, relationships, and past experiences.

When memories are provided, use them naturally to inform your responses. Make connections between different memories when relevant.

Think step by step in <think>...</think> tags before responding.

Be warm and helpful, like a thoughtful friend who knows them well.

Important: Do not use emojis in your responses.""",
    memory=MemoryConfig(
        memory_file="/app/dad_memories.json",
        memory_prefix="[User's memories and notes:]",
    ),
)

