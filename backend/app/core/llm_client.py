"""
LLM Client - Unified interface for LLM providers.

Supports:
- Anthropic Claude (primary)
- OpenAI GPT (alternative)
- Local fallback response
"""

import logging
import os
from typing import AsyncGenerator

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified async LLM client."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        self._client = None

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate a response from the configured LLM."""
        if self.provider == "anthropic":
            return await self._generate_anthropic(messages, max_tokens, temperature)
        elif self.provider == "openai":
            return await self._generate_openai(messages, max_tokens, temperature)
        else:
            return await self._generate_anthropic(messages, max_tokens, temperature)

    async def generate_stream(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation - yields text chunks."""
        if self.provider == "anthropic":
            async for chunk in self._stream_anthropic(messages, max_tokens, temperature):
                yield chunk
        else:
            # Fallback: non-streaming
            response = await self.generate(messages, max_tokens, temperature)
            yield response

    async def _generate_anthropic(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> str:
        """Generate response using Anthropic API."""
        try:
            import anthropic

            api_key = settings.ANTHROPIC_API_KEY
            if not api_key:
                logger.warning("No Anthropic API key found, using mock response")
                return self._mock_response(messages)

            client = anthropic.AsyncAnthropic(api_key=api_key)

            # Separate system message
            system_content = ""
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    chat_messages.append(msg)

            response = await client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_content,
                messages=chat_messages,
            )
            return response.content[0].text

        except ImportError:
            logger.warning("anthropic package not installed. Using OpenAI fallback.")
            return await self._generate_openai(messages, max_tokens, temperature)
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"I encountered an error while processing your request. Please try again. (Error: {str(e)[:100]})"

    async def _stream_anthropic(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> AsyncGenerator[str, None]:
        """Streaming response from Anthropic."""
        try:
            import anthropic

            api_key = settings.ANTHROPIC_API_KEY
            if not api_key:
                yield self._mock_response(messages)
                return

            client = anthropic.AsyncAnthropic(api_key=api_key)

            system_content = ""
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    chat_messages.append(msg)

            async with client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_content,
                messages=chat_messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            yield f"Error generating response: {str(e)[:100]}"

    async def _generate_openai(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> str:
        """Generate response using OpenAI API."""
        try:
            from openai import AsyncOpenAI

            api_key = settings.OPENAI_API_KEY
            if not api_key:
                return self._mock_response(messages)

            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            response = await client.chat.completions.create(
                model=self.model or "gpt-3.5-turbo",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        except ImportError:
            return self._mock_response(messages)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error: {str(e)[:100]}"

    def _mock_response(self, messages: list[dict]) -> str:
        """Mock response for testing without an API key."""
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return (
            f"[Demo Mode - No API Key Configured]\n\n"
            f"I received your question: '{last_user[:100]}'\n\n"
            f"To enable real AI responses, please set the ANTHROPIC_API_KEY or OPENAI_API_KEY "
            f"environment variable in your deployment configuration."
        )
