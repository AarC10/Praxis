from unittest.mock import MagicMock, patch

import pytest

from praxis.executor import ExecutionConfig, SkillExecutor
from praxis.refiner import RepairResult
from praxis.sandbox import SandboxResult
from praxis.skill import ExecutionContext, ExecutionResult, Skill


def _make_skill(**overrides) -> Skill:
    defaults = dict(
        skill_id="test_skill",
        name="test_skill",
        short_desc="Test",
        version_id="test_skill_v1",
        version=1,
        code="def run(context):\n    return {'out': 1}",
        code_path="skills/test_skill_v1.py",
        checksum="",
        contract_name="standalone_python",
        inputs_schema={"x": "int"},
        outputs_schema={"out": "int"},
    )
    defaults.update(overrides)
    return Skill(**defaults)


def _make_context(**overrides) -> ExecutionContext:
    defaults = dict(
        user_request="test",
        params={"x": 1},
    )
    defaults.update(overrides)
    return ExecutionContext(**defaults)


def _ok_result(**overrides) -> SandboxResult:
    defaults = dict(success=True, data={"out": 1}, latency_ms=10)
    defaults.update(overrides)
    return SandboxResult(**defaults)


def _fail_result(**overrides) -> SandboxResult:
    defaults = dict(success=False, error="NameError: bad", latency_ms=5)
    defaults.update(overrides)
    return SandboxResult(**defaults)


@pytest.fixture()
def executor_no_llm():
    return SkillExecutor(
        llm_client=None,
        storage=None,
        config=ExecutionConfig(max_repair_attempts=0, max_retries=0),
    )


class TestSuccessfulExecution:
    @patch("praxis.executor.run_in_sandbox")
    def test_returns_success_result(self, mock_sandbox, executor_no_llm):
        mock_sandbox.return_value = _ok_result()
        result = executor_no_llm.execute(_make_skill(), _make_context())

        assert result.success
        assert result.data == {"out": 1}
        assert result.latency_ms == 10
        assert result.exec_id is not None

    @patch("praxis.executor.run_in_sandbox")
    def test_sandbox_receives_context_params(self, mock_sandbox, executor_no_llm):
        mock_sandbox.return_value = _ok_result()
        executor_no_llm.execute(_make_skill(), _make_context(params={"x": 42}))

        _, kwargs = mock_sandbox.call_args
        assert kwargs["context_data"]["params"] == {"x": 42}


class TestInputValidation:
    @patch("praxis.executor.run_in_sandbox")
    def test_missing_required_param_fails_without_sandbox(self, mock_sandbox, executor_no_llm):
        result = executor_no_llm.execute(
            _make_skill(inputs_schema={"x": "int", "y": "int"}),
            _make_context(params={"x": 1}),  # y is missing
        )

        assert not result.success
        assert "y" in result.error
        mock_sandbox.assert_not_called()

class TestUnknownContract:
    @patch("praxis.executor.run_in_sandbox")
    def test_unknown_contract_fails_without_sandbox(self, mock_sandbox, executor_no_llm):
        skill = _make_skill(contract_name="does_not_exist")
        result = executor_no_llm.execute(skill, _make_context())

        assert not result.success
        assert "does_not_exist" in result.error
        mock_sandbox.assert_not_called()


class TestContractViolations:
    # validate_contract is imported inline inside execute(), so patch
    # the function in its home module praxis.sandbox instead of praxis.executor
    @patch("praxis.sandbox.validate_contract", return_value=["Forbidden import: os"])
    @patch("praxis.executor.run_in_sandbox")
    def test_static_violation_fails_when_no_llm(self, mock_sandbox, mock_validate,
                                                executor_no_llm):
        result = executor_no_llm.execute(_make_skill(), _make_context())

        assert not result.success
        assert "Forbidden import" in result.error
        mock_sandbox.assert_not_called()

    @patch("praxis.sandbox.validate_contract", return_value=["Forbidden import: os"])
    @patch("praxis.executor.run_in_sandbox")
    @patch("praxis.executor.repair_skill")
    def test_static_violation_repaired_by_llm(self, mock_repair, mock_sandbox, mock_validate):
        mock_repair.return_value = RepairResult(
            success=True, repaired_code="def run(context): return {'out': 1}", attempts=1
        )
        # After repair, validate_contract must clear
        mock_sandbox.return_value = _ok_result()

        executor = SkillExecutor(
            llm_client=MagicMock(),
            storage=None,
            config=ExecutionConfig(max_repair_attempts=1, max_retries=0),
        )
        result = executor.execute(_make_skill(), _make_context())

        assert result.success
        mock_repair.assert_called_once()

    @patch("praxis.sandbox.validate_contract", return_value=["Forbidden import: os"])
    @patch("praxis.executor.run_in_sandbox")
    @patch("praxis.executor.repair_skill")
    def test_static_violation_repair_fails_returns_error(self, mock_repair, mock_sandbox,
                                                         mock_validate):
        mock_repair.return_value = RepairResult(
            success=False, error="Could not fix", attempts=1
        )

        executor = SkillExecutor(
            llm_client=MagicMock(),
            storage=None,
            config=ExecutionConfig(max_repair_attempts=1, max_retries=0),
        )
        result = executor.execute(_make_skill(), _make_context())

        assert not result.success
        mock_sandbox.assert_not_called()


class TestRepairLoop:
    @patch("praxis.executor.run_in_sandbox")
    def test_failure_no_llm_returns_error(self, mock_sandbox, executor_no_llm):
        mock_sandbox.return_value = _fail_result(error="NameError: bad")
        result = executor_no_llm.execute(_make_skill(), _make_context())

        assert not result.success
        assert "NameError" in result.error

    @patch("praxis.executor.repair_skill")
    @patch("praxis.executor.run_in_sandbox")
    def test_failure_repaired_on_second_attempt(self, mock_sandbox, mock_repair):
        mock_sandbox.side_effect = [
            _fail_result(error="NameError"),
            _ok_result(),
        ]
        mock_repair.return_value = RepairResult(
            success=True, repaired_code="def run(context): return {'out': 1}", attempts=1
        )

        executor = SkillExecutor(
            llm_client=MagicMock(),
            storage=None,
            config=ExecutionConfig(max_repair_attempts=1, max_retries=0),
        )
        result = executor.execute(_make_skill(), _make_context())

        assert result.success
        assert mock_sandbox.call_count == 2
        mock_repair.assert_called_once()

    @patch("praxis.executor.repair_skill")
    @patch("praxis.executor.run_in_sandbox")
    def test_repair_exhausted_returns_last_error(self, mock_sandbox, mock_repair):
        mock_sandbox.return_value = _fail_result(error="always fails")
        mock_repair.return_value = RepairResult(
            success=True, repaired_code="def run(context): return {'out': 0}", attempts=1
        )

        executor = SkillExecutor(
            llm_client=MagicMock(),
            storage=None,
            config=ExecutionConfig(max_repair_attempts=2, max_retries=0),
        )
        result = executor.execute(_make_skill(), _make_context())

        assert not result.success
        assert "always fails" in result.error
        # sandbox called once per repair round (3 total: initial + 2 repairs)
        assert mock_sandbox.call_count == 3

    @patch("praxis.executor.repair_skill")
    @patch("praxis.executor.run_in_sandbox")
    def test_repair_failure_stops_loop(self, mock_sandbox, mock_repair):
        """If the LLM can't produce a repair, don't keep trying."""
        mock_sandbox.return_value = _fail_result(error="some error")
        mock_repair.return_value = RepairResult(success=False, error="LLM failed", attempts=1)

        executor = SkillExecutor(
            llm_client=MagicMock(),
            storage=None,
            config=ExecutionConfig(max_repair_attempts=3, max_retries=0),
        )
        result = executor.execute(_make_skill(), _make_context())

        assert not result.success
        # Only one sandbox call (initial) then repair failed → stop
        assert mock_sandbox.call_count == 1
        mock_repair.assert_called_once()


class TestTimeoutHandling:
    @patch("praxis.executor.repair_skill")
    @patch("praxis.executor.run_in_sandbox")
    def test_timeout_not_repaired(self, mock_sandbox, mock_repair):
        mock_sandbox.return_value = _fail_result(
            error="Skill exceeded timeout of 30s", latency_ms=30000
        )

        executor = SkillExecutor(
            llm_client=MagicMock(),
            storage=None,
            config=ExecutionConfig(max_repair_attempts=3, max_retries=0),
        )
        result = executor.execute(_make_skill(), _make_context())

        assert not result.success
        assert "timeout" in result.error.lower()
        mock_repair.assert_not_called()


class TestOutputSchemaValidation:
    @patch("praxis.executor.run_in_sandbox")
    def test_output_mismatch_logged_not_failed(self, mock_sandbox, executor_no_llm):
        """Missing output keys produce a warning but the result is still Success."""
        mock_sandbox.return_value = _ok_result(data={"wrong_key": 1})
        skill = _make_skill(outputs_schema={"out": "int"})

        result = executor_no_llm.execute(skill, _make_context())

        # Execution is still considered successful — the caller gets the raw data
        assert result.success
        assert result.data == {"wrong_key": 1}


class TestExecutionLogging:
    @patch("praxis.executor.run_in_sandbox")
    def test_execution_logged_to_storage_on_success(self, mock_sandbox):
        mock_sandbox.return_value = _ok_result()
        mock_storage = MagicMock()
        mock_storage.db.execute = MagicMock()
        mock_storage.db.commit = MagicMock()

        executor = SkillExecutor(
            llm_client=None,
            storage=mock_storage,
            config=ExecutionConfig(max_repair_attempts=0, max_retries=0),
        )
        result = executor.execute(_make_skill(), _make_context())

        assert result.success
        mock_storage.db.execute.assert_called()
        mock_storage.db.commit.assert_called()

    @patch("praxis.executor.run_in_sandbox")
    def test_execution_logged_to_storage_on_failure(self, mock_sandbox):
        mock_sandbox.return_value = _fail_result()
        mock_storage = MagicMock()
        mock_storage.db.execute = MagicMock()
        mock_storage.db.commit = MagicMock()

        executor = SkillExecutor(
            llm_client=None,
            storage=mock_storage,
            config=ExecutionConfig(max_repair_attempts=0, max_retries=0),
        )
        result = executor.execute(_make_skill(), _make_context())

        assert not result.success
        mock_storage.db.execute.assert_called()
