"""LLM provider using LiteLLM for multi-provider support."""

import json
from enum import Enum
from typing import Optional

import litellm
from pydantic import BaseModel


class Role(str, Enum):
    """Message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Chat message."""

    role: Role
    content: str


class LLMConfig(BaseModel):
    """LLM configuration."""

    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class LLMProvider:
    """Multi-provider LLM interface via LiteLLM."""

    def __init__(self, config: LLMConfig):
        self.config = config
        if config.api_key:
            litellm.api_key = config.api_key
        if config.api_base:
            litellm.api_base = config.api_base

    async def complete(self, messages: list[Message]) -> str:
        """Get completion from LLM."""
        formatted = [{"role": m.role.value, "content": m.content} for m in messages]

        response = await litellm.acompletion(
            model=self.config.model,
            messages=formatted,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return response.choices[0].message.content

    async def complete_json(self, messages: list[Message]) -> dict:
        """Get JSON completion from LLM."""
        formatted = [{"role": m.role.value, "content": m.content} for m in messages]

        response = await litellm.acompletion(
            model=self.config.model,
            messages=formatted,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)
