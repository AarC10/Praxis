import os
import logging
import requests
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

DEFAULT_GENERATION_MODEL = "claude-sonnet-4-5-20250929"

# TODO: Maybe just pay voyage?
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

class AnthropicClient(LLMClient):
    def __init__(
        self,
        model: str = DEFAULT_GENERATION_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        max_retries: int = 3,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.max_retries = max_retries

        try:
            import anthropic
        except ImportError:
            raise LLMClientError("anthropic package not installed. Run: pip install anthropic")

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            raise LLMAuthenticationError("ANTHROPIC_API_KEY environment variable not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._anthropic = anthropic

        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Connected to Ollama at {self.ollama_base_url} for embeddings")
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Cannot connect to Ollama at {self.ollama_base_url}. "
                "Embeddings will fail. Make sure Ollama is running: ollama serve"
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama connection test failed: {e}")

        logger.info(f"Initialized AnthropicClient with model={model}")

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        max_tokens = kwargs.get("max_tokens", 4096)
        temperature = kwargs.get("temperature", 0.7)
        system = kwargs.get("system", None)
        def _make_request():
            request_kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if system:
                request_kwargs["system"] = system
            return self._client.messages.create(**request_kwargs)
        retryable = (
            self._anthropic.RateLimitError,
            self._anthropic.APIConnectionError,
            self._anthropic.InternalServerError,
        )
        try:
            response, latency_ms = time_request(
                lambda: retry_with_backoff(_make_request, self.max_retries, retryable_exceptions=retryable)
            )
        except self._anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(f"Anthropic authentication failed: {e}")
        except self._anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {e}")
        except self._anthropic.APIConnectionError as e:
            raise LLMConnectionError(f"Anthropic connection error: {e}")
        except Exception as e:
            raise LLMClientError(f"Anthropic generation failed: {e}")
        content = "".join(block.text for block in response.content if hasattr(block, "text"))
        llm_response = LLMResponse(
            content=content,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            latency_ms=latency_ms,
            model=response.model,
        )
        logger.info(f"Generated {llm_response.tokens_output} tokens in {latency_ms}ms")
        return llm_response

    def embed(self, text: str) -> bytes:
        def _make_request():
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["embedding"]

        retryable = (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        )
        try:
            embedding, latency_ms = time_request(
                lambda: retry_with_backoff(_make_request, self.max_retries, retryable_exceptions=retryable)
            )
        except requests.exceptions.HTTPError as e:
            raise LLMClientError(f"Ollama HTTP error: {e}")
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Ollama connection error: {e}")
        except Exception as e:
            raise LLMClientError(f"Ollama embedding failed: {e}")
        logger.info(f"Created embedding with {len(embedding)} dimensions in {latency_ms}ms")
        return embedding_to_bytes(embedding)

    def get_model_name(self) -> str:
        return self.model
