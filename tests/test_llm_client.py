import pytest
import struct
import time
from unittest.mock import Mock

from praxis.llm_client import (
    LLMClient,
    LLMResponse,
    LLMClientError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMConnectionError,
    embedding_to_bytes,
    bytes_to_embedding,
    retry_with_backoff,
    time_request,
)


class TestLLMResponse:
    def test_basic_creation(self):
        response = LLMResponse(
            content="def hello(): pass",
            tokens_input=100,
            tokens_output=50,
            latency_ms=500,
            model="test-model"
        )
        assert response.content == "def hello(): pass"
        assert response.tokens_input == 100
        assert response.tokens_output == 50
        assert response.latency_ms == 500
        assert response.model == "test-model"

    def test_total_tokens(self):
        response = LLMResponse(
            content="test",
            tokens_input=100,
            tokens_output=50,
            latency_ms=500,
            model="test"
        )
        assert response.total_tokens == 150

    def test_to_dict(self):
        response = LLMResponse(
            content="code",
            tokens_input=10,
            tokens_output=20,
            latency_ms=100,
            model="model"
        )
        d = response.to_dict()
        assert d["content"] == "code"
        assert d["tokens_input"] == 10
        assert d["tokens_output"] == 20
        assert d["latency_ms"] == 100
        assert d["model"] == "model"
        assert d["total_tokens"] == 30


class TestEmbeddingConversion:
    def test_embedding_to_bytes(self):
        embedding = [0.1, 0.2, 0.3]
        result = embedding_to_bytes(embedding)
        assert isinstance(result, bytes)
        assert len(result) == 12

    def test_bytes_to_embedding(self):
        floats = [1.0, 2.0, 3.0]
        data = struct.pack("3f", *floats)
        result = bytes_to_embedding(data)
        assert len(result) == 3
        assert all(abs(a - b) < 0.0001 for a, b in zip(floats, result))

    def test_round_trip(self):
        original = [0.1, -0.5, 1.0, 0.0, -1.0]
        encoded = embedding_to_bytes(original)
        decoded = bytes_to_embedding(encoded)
        assert len(original) == len(decoded)
        for a, b in zip(original, decoded):
            assert abs(a - b) < 0.0001

    def test_empty_embedding(self):
        result = embedding_to_bytes([])
        assert result == b""
        assert bytes_to_embedding(b"") == []

    def test_large_embedding(self):
        original = [float(i) / 1000 for i in range(1536)]
        encoded = embedding_to_bytes(original)
        assert len(encoded) == 1536 * 4
        decoded = bytes_to_embedding(encoded)
        assert len(decoded) == 1536


class TestRetryWithBackoff:
    def test_success_first_try(self):
        func = Mock(return_value="success")
        result = retry_with_backoff(func, max_retries=3)
        assert result == "success"
        assert func.call_count == 1

    def test_success_after_retry(self):
        func = Mock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])
        result = retry_with_backoff(
            func,
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ValueError,)
        )
        assert result == "success"
        assert func.call_count == 3

    def test_exhausted_retries(self):
        func = Mock(side_effect=ValueError("always fails"))
        with pytest.raises(ValueError, match="always fails"):
            retry_with_backoff(
                func,
                max_retries=2,
                base_delay=0.01,
                retryable_exceptions=(ValueError,)
            )
        assert func.call_count == 3

    def test_non_retryable_exception(self):
        func = Mock(side_effect=TypeError("not retryable"))
        with pytest.raises(TypeError):
            retry_with_backoff(
                func,
                max_retries=3,
                retryable_exceptions=(ValueError,)
            )
        assert func.call_count == 1


class TestTimeRequest:
    def test_returns_result_and_time(self):
        func = Mock(return_value="result")
        result, latency_ms = time_request(func)
        assert result == "result"
        assert isinstance(latency_ms, int)
        assert latency_ms >= 0

    def test_timing_accuracy(self):
        def slow_func():
            time.sleep(0.1)
            return "done"

        result, latency_ms = time_request(slow_func)
        assert result == "done"
        assert latency_ms >= 100
        assert latency_ms < 200


class TestLLMClientAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            LLMClient()

    def test_concrete_implementation_works(self):
        class ConcreteClient(LLMClient):
            def generate(self, prompt: str, **kwargs) -> LLMResponse:
                return LLMResponse(
                    content=f"Generated from: {prompt}",
                    tokens_input=len(prompt),
                    tokens_output=10,
                    latency_ms=100,
                    model="test"
                )

            def embed(self, text: str) -> bytes:
                return embedding_to_bytes([0.1, 0.2, 0.3])

            def get_model_name(self) -> str:
                return "test-model"

        client = ConcreteClient()
        response = client.generate("test prompt")
        assert response.content == "Generated from: test prompt"
        assert client.get_model_name() == "test-model"

        embedding = client.embed("test text")
        assert isinstance(embedding, bytes)


class TestExceptions:
    def test_exception_hierarchy(self):
        assert issubclass(LLMRateLimitError, LLMClientError)
        assert issubclass(LLMAuthenticationError, LLMClientError)
        assert issubclass(LLMConnectionError, LLMClientError)

    def test_exception_messages(self):
        error = LLMClientError("test message")
        assert str(error) == "test message"

        rate_error = LLMRateLimitError("rate limit hit")
        assert str(rate_error) == "rate limit hit"
