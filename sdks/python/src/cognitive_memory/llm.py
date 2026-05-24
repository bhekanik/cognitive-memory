"""
LLM provider abstraction.

The SDK calls LLMs in three places (memory extraction, conflict detection,
consolidation summary, optional rerank). Rather than hard-wiring those calls
to OpenAI, the engine talks to an :class:`LLMProvider` interface; callers can
inject any backend that satisfies the contract.

The default provider :class:`OpenAILLMProvider` preserves the SDK's prior
behaviour (OpenAI client created lazily, retries on transient errors) so
existing code that does not pass a provider keeps working.
"""

from __future__ import annotations

import time
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, TypedDict


class LLMUsage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int


class LLMProvider(ABC):
    """Pluggable text-completion backend."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Return the completion text for ``prompt``."""

    def complete_with_usage(
        self, prompt: str, **kwargs: Any
    ) -> tuple[str, LLMUsage]:
        """Return (text, usage) where ``usage`` carries token counts.

        Default implementation calls :meth:`complete` and reports zero usage.
        Providers that can supply token counts SHOULD override.
        """
        return self.complete(prompt, **kwargs), {}


class OpenAILLMProvider(LLMProvider):
    """Default provider. Wraps the OpenAI Python client.

    Retries transient errors (HTTP 5xx, rate limits, network blips) with
    exponential backoff. The OpenAI client is created lazily so importing this
    module does not require ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 5,
        timeout: float = 120.0,
    ):
        self._model = model
        self._max_retries = max_retries
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"timeout": self._timeout}
            chat_base_url = os.getenv("OPENAI_CHAT_BASE_URL")
            chat_api_key = os.getenv("OPENAI_CHAT_API_KEY")
            chat_timeout = os.getenv("OPENAI_CHAT_TIMEOUT")
            if chat_timeout:
                kwargs["timeout"] = float(chat_timeout)
            if chat_base_url:
                kwargs["base_url"] = chat_base_url
                kwargs["api_key"] = chat_api_key or "local-chat"
            elif chat_api_key:
                kwargs["api_key"] = chat_api_key
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(self, prompt: str, **kwargs: Any) -> str:
        text, _ = self.complete_with_usage(prompt, **kwargs)
        return text

    def complete_with_usage(
        self, prompt: str, **kwargs: Any
    ) -> tuple[str, LLMUsage]:
        max_tokens: Optional[int] = kwargs.get("max_tokens")
        model: str = kwargs.get("model") or self._model
        temperature: float = kwargs.get("temperature", 0)

        client = self._get_client()
        for attempt in range(self._max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content.strip()
                usage: LLMUsage = {}
                if getattr(resp, "usage", None) is not None:
                    usage = {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(
                            resp.usage, "completion_tokens", 0
                        )
                        or 0,
                    }
                return text, usage
            except Exception as e:
                err_str = str(e).lower()
                retryable = any(
                    k in err_str
                    for k in (
                        "500",
                        "server_error",
                        "502",
                        "503",
                        "529",
                        "rate_limit",
                        "timeout",
                        "connection",
                    )
                )
                if attempt < self._max_retries - 1 and retryable:
                    time.sleep(min(60, 2**attempt * 2))
                    continue
                raise

        # Unreachable; the loop either returns or raises.
        raise RuntimeError("OpenAILLMProvider: exhausted retries without raising")
