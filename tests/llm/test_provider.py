"""Tests for LLM provider."""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from agent_db.llm.provider import LLMProvider, LLMConfig, Message, Role


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
    )


class TestMessage:
    def test_message_creation(self):
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"


class TestLLMProvider:
    @pytest.mark.asyncio
    async def test_completion(self, llm_config: LLMConfig):
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock_choice = MagicMock()
            mock_choice.message.content = "Hi!"
            mock.return_value.choices = [mock_choice]

            provider = LLMProvider(llm_config)
            messages = [Message(role=Role.USER, content="Hello")]
            result = await provider.complete(messages)

            assert result == "Hi!"
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_completion_json(self, llm_config: LLMConfig):
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock_choice = MagicMock()
            mock_choice.message.content = '{"key": "value"}'
            mock.return_value.choices = [mock_choice]

            provider = LLMProvider(llm_config)
            messages = [Message(role=Role.USER, content="Return JSON")]
            result = await provider.complete_json(messages)

            assert result == {"key": "value"}
