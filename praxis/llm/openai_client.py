import os
import logging
from typing import Optional
from praxis.llm_client import (
    LLMClient,
    LLMResponse,
    LLMClientError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMConnectionError,
    embedding_to_bytes,
    retry_with_backoff,
    time_request,
)

logger = logging.getLogger(__name__)

DEFAULT_GENERATION_MODEL = "gpt-5.2"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str = DEFAULT_GENERATION_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        max_retries: int = 3,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.max_retries = max_retries

        try:
            import openai
        except ImportError:
            raise LLMClientError("openai package not installed. Run: pip install openai")

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMAuthenticationError("OPENAI_API_KEY environment variable not set")

        self._client = openai.OpenAI(api_key=api_key)
        self._openai = openai

        logger.info(f"Initialized OpenAIClient with model={model}")

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        max_tokens = kwargs.get("max_tokens", 4096)
        temperature = kwargs.get("temperature", 0.7)
        system = kwargs.get("system", None)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def _make_request():
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        retryable = (
            self._openai.RateLimitError,
            self._openai.APIConnectionError,
            self._openai.InternalServerError,
        )

        try:
            response, latency_ms = time_request(
                lambda: retry_with_backoff(_make_request, self.max_retries, retryable_exceptions=retryable)
            )
        except self._openai.AuthenticationError as e:
            raise LLMAuthenticationError(f"OpenAI authentication failed: {e}")
        except self._openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}")
        except self._openai.APIConnectionError as e:
            raise LLMConnectionError(f"OpenAI connection error: {e}")
        except Exception as e:
            raise LLMClientError(f"OpenAI generation failed: {e}")

        content = response.choices[0].message.content or ""
        llm_response = LLMResponse(
            content=content,
            tokens_input=response.usage.prompt_tokens,
            tokens_output=response.usage.completion_tokens,
            latency_ms=latency_ms,
            model=response.model,
        )

        logger.info(f"Generated {llm_response.tokens_output} tokens in {latency_ms}ms")
        return llm_response

    def embed(self, text: str) -> bytes:
        def _make_request():
            response = self._client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return response.data[0].embedding
        retryable = (
            self._openai.RateLimitError,
            self._openai.APIConnectionError,
            self._openai.InternalServerError,
        )
        try:
            embedding, latency_ms = time_request(
                lambda: retry_with_backoff(_make_request, self.max_retries, retryable_exceptions=retryable)
            )
        except Exception as e:
            raise LLMClientError(f"OpenAI embedding failed: {e}")
        logger.info(f"Created embedding with {len(embedding)} dimensions in {latency_ms}ms")
        return embedding_to_bytes(embedding)

    def get_model_name(self) -> str:
        return self.model
