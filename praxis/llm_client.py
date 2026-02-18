from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import struct
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output

    def to_dict(self) -> dict[str, Any]:
        # Storage pruposes later
        return {
            "content": self.content,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "total_tokens": self.total_tokens,
        }


class LLMClient(ABC):
    """
    Impls must handle:
    - API initialization from environment variables
    - Exponential backoff retry logic
    - Request timing and token counting
    - Response normalization to LLMResponse format
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text from a prompt. SHould request py code for implementing skill

        @param prompt: The input prompt (skill generation request)
        @param kwargs: Provider specific parameters (temperature, max_tokens, etc.)
        @return LLMResponse with generated content and metrics
        @exception LLMClientError: If generation fails after retries
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> bytes:
        """
        Create an embedding vector for text.

        Used for semantic search to find similar existing skills in the database
        Embedding quality directly will impact search results so maybe have this be provider specific?

        @param text The text to embed (skill descriptions, user queries)
        @return Embedding as bytes (float array packed with struct.pack)
        @exception LLMClientError If embedding fails after retries
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        # self explanatory
        pass


class LLMClientError(Exception):
    pass


class LLMRateLimitError(LLMClientError):
    pass


class LLMAuthenticationError(LLMClientError):
    pass


class LLMConnectionError(LLMClientError):
    pass


def embedding_to_bytes(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def bytes_to_embedding(data: bytes) -> list[float]:
    num_floats = len(data) // 4  # 4 bytes per float
    return list(struct.unpack(f"{num_floats}f", data))


def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
):
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(f"All {max_retries} retries exhausted. Last error: {e}")
                raise

            delay = min(base_delay * (exponential_base**attempt), max_delay)
            logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. " f"Retrying in {delay:.1f}s...")
            time.sleep(delay)

    raise last_exception


def time_request(func) -> tuple[Any, int]:
    start = time.perf_counter()
    result = func()
    end = time.perf_counter()
    latency_ms = int((end - start) * 1000)
    return result, latency_ms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM clients with code generation")
    parser.add_argument(
        "-p",
        "--provider",
        choices=["openai", "anthropic", "ollama"],
        default="ollama",
        help="LLM provider to use (default: ollama)",
    )
    parser.add_argument("-m", "--model", default=None, help="Model name (uses provider default if not specified)")
    parser.add_argument(
        "--prompt",
        default="Write a Python function that calculates the fibonacci sequence up to n terms",
        help="Code generation prompt",
    )
    args = parser.parse_args()

    llm_client: LLMClient
    if args.provider == "openai":
        from praxis.llm.openai_client import OpenAIClient

        llm_client = OpenAIClient(model=args.model) if args.model else OpenAIClient()
    elif args.provider == "anthropic":
        from praxis.llm.anthropic_client import AnthropicClient

        llm_client = AnthropicClient(model=args.model) if args.model else AnthropicClient()
    elif args.provider == "ollama":
        from praxis.llm.ollama_client import OllamaClient

        llm_client = OllamaClient(model=args.model) if args.model else OllamaClient()
    else:
        print(f"Unknown provider: {args.provider}")
        exit(1)

    print(f"Using provider: {args.provider}")
    print(f"Using model: {llm_client.get_model_name()}")
    print(f"\n{"="*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{"="*60}\n")

    try:
        response = llm_client.generate(
            args.prompt,
            system="As a coding assistant, generate clean, working Python code.",
            temperature=0.7,
        )

        print("Generated Code:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print(f"\nMetrics:")
        print(f"  Model: {response.model}")
        print(f"  Input tokens: {response.tokens_input}")
        print(f"  Output tokens: {response.tokens_output}")
        print(f"  Total tokens: {response.total_tokens}")
        print(f"  Latency: {response.latency_ms}ms")

    except LLMClientError as e:
        print(f"Error: {e}")
