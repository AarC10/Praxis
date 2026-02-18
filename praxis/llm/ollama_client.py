import logging

import requests

from praxis.llm_client import (
    LLMClient,
    LLMResponse,
    LLMClientError,
    LLMConnectionError,
    embedding_to_bytes,
    retry_with_backoff,
    time_request,
)

logger = logging.getLogger(__name__)

DEFAULT_GENERATION_MODEL = "qwen2.5-coder:7b"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str = DEFAULT_GENERATION_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. " "Make sure Ollama is running: ollama serve"
            )
        except requests.exceptions.RequestException as e:
            raise LLMConnectionError(f"Ollama connection test failed: {e}")
        logger.info(f"Initialized OllamaClient with model={model}, base_url={base_url}")

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        temperature = kwargs.get("temperature", 0.7)
        system = kwargs.get("system", None)
        num_predict = kwargs.get("max_tokens", 4096)

        def _make_request():
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
            }
            if system:
                payload["system"] = system
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        retryable = (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        )
        try:
            response_data, latency_ms = time_request(
                lambda: retry_with_backoff(_make_request, self.max_retries, retryable_exceptions=retryable)
            )
        except requests.exceptions.HTTPError as e:
            raise LLMClientError(f"Ollama HTTP error: {e}")
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Ollama connection error: {e}")
        except Exception as e:
            raise LLMClientError(f"Ollama generation failed: {e}")
        content = response_data.get("response", "")
        prompt_eval_count = response_data.get("prompt_eval_count", 0)
        eval_count = response_data.get("eval_count", 0)
        llm_response = LLMResponse(
            content=content,
            tokens_input=prompt_eval_count,
            tokens_output=eval_count,
            latency_ms=latency_ms,
            model=self.model,
        )
        logger.info(f"Generated {llm_response.tokens_output} tokens in {latency_ms}ms")
        return llm_response

    def embed(self, text: str) -> bytes:
        def _make_request():
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text,
                },
                timeout=self.timeout,
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
        except Exception as e:
            raise LLMClientError(f"Ollama embedding failed: {e}")
        logger.info(f"Created embedding with {len(embedding)} dimensions in {latency_ms}ms")
        return embedding_to_bytes(embedding)

    def get_model_name(self) -> str:
        return self.model

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=600,
                stream=True,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    logger.debug(f"Pull progress: {line.decode()}")
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
