import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from praxis.contracts import CONTRACTS, SkillContract
from praxis.llm_client import LLMClient
from praxis.refiner import repair_skill, RepairResult
from praxis.sandbox import (
    run_in_sandbox,
    SandboxError,
    SandboxTimeoutError,
    ContractViolationError,
    SandboxResult,
)
from praxis.skill import ExecutionContext, ExecutionResult, Skill

logger = logging.getLogger(__name__)

_TRANSIENT_SANDBOX_ERRORS = (
    OSError,
    PermissionError,
)


@dataclass
class ExecutionConfig:
    max_repair_attempts: int = 3
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    save_repaired_versions: bool = True


class SkillExecutor:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        storage=None,
        config: Optional[ExecutionConfig] = None,
    ):
        """
        Args:
            llm_client: Required for auto-repair. If None, repair is disabled.
            storage: SkillStorage instance for logging executions and saving repaired versions.
            config: Execution configuration.
        """
        self.llm_client = llm_client
        self.storage = storage
        self.config = config or ExecutionConfig()

    def execute(
        self,
        skill: Skill,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute a skill with full retry/repair/logging pipeline.

        Args:
            skill: The skill to execute
            context: Execution context with params and metadata

        Returns:
            ExecutionResult — always returns, never raises (errors are captured)
        """
        exec_id = str(uuid.uuid4())
        start_time = datetime.now()

        logger.info(f"[{exec_id}] Executing skill '{skill.name}' v{skill.version}")

        # 1. Input validation
        is_valid, schema_errors = skill.matches_input_schema(context.params)
        if not is_valid:
            error_msg = f"Input schema validation failed: {'; '.join(schema_errors)}"
            logger.warning(f"[{exec_id}] {error_msg}")
            result = ExecutionResult(
                success=False,
                error=error_msg,
                exec_id=exec_id,
                latency_ms=0,
            )
            self._log_execution(exec_id, skill, context, result, start_time)
            return result

        # contract
        contract = CONTRACTS.get(skill.contract_name)
        if not contract:
            error_msg = f"Unknown contract: {skill.contract_name}"
            logger.error(f"[{exec_id}] {error_msg}")
            result = ExecutionResult(success=False, error=error_msg, exec_id=exec_id, latency_ms=0)
            self._log_execution(exec_id, skill, context, result, start_time)
            return result

        # Contract static check (before we  burn a subprocess!!)
        from praxis.sandbox import validate_contract
        violations = validate_contract(skill.code, contract)
        if violations:
            error_msg = f"Contract violations: {'; '.join(violations)}"
            logger.error(f"[{exec_id}] {error_msg}")

            # Try to repair the violations before giving up
            if self.llm_client:
                repair = self._attempt_repair(
                    skill, error_msg, "", context.params, contract
                )
                if repair.success:
                    skill = self._apply_repair(skill, repair.repaired_code)
                else:
                    result = ExecutionResult(
                        success=False, error=repair.error, exec_id=exec_id, latency_ms=0
                    )
                    self._log_execution(exec_id, skill, context, result, start_time)
                    return result
            else:
                result = ExecutionResult(success=False, error=error_msg, exec_id=exec_id, latency_ms=0)
                self._log_execution(exec_id, skill, context, result, start_time)
                return result

        # Main execution loop with repair
        context_data = {
            "user_request": context.user_request,
            "params": context.params,
            "metadata": context.metadata,
            "llm_model": context.llm_model,
            "timeout_seconds": context.timeout_seconds,
        }

        current_skill_code = skill.code
        total_latency_ms = 0
        last_error = None
        last_traceback = None

        repair_attempts_used = 0
        max_repairs = self.config.max_repair_attempts if self.llm_client else 0

        for repair_round in range(max_repairs + 1):
            sandbox_result = self._run_with_retry(
                current_skill_code,
                context_data,
                contract,
                context.timeout_seconds,
            )
            total_latency_ms += sandbox_result.latency_ms

            if sandbox_result.success:
                # Validate outputs match the schema
                output_valid, output_errors = skill.validates_output(sandbox_result.data or {})
                if not output_valid:
                    logger.warning(
                        f"[{exec_id}] Output schema mismatch: {output_errors}. "
                        "Returning result anyway."
                    )

                result = ExecutionResult(
                    success=True,
                    data=sandbox_result.data,
                    exec_id=exec_id,
                    latency_ms=total_latency_ms,
                )

                # If we used repaired code, persist it
                if repair_round > 0 and self.config.save_repaired_versions and self.storage:
                    self._save_repaired_skill(skill, current_skill_code)

                self._log_execution(exec_id, skill, context, result, start_time)
                logger.info(
                    f"[{exec_id}] Success in {total_latency_ms}ms "
                    f"(repair rounds: {repair_round})"
                )
                return result

            # Execution failed
            last_error = sandbox_result.error
            last_traceback = sandbox_result.traceback

            logger.warning(
                f"[{exec_id}] Execution failed (round {repair_round}): {last_error}"
            )

            # Out of repair attempts?
            if repair_round >= max_repairs or not self.llm_client:
                break

            # Timeout is not repairable (it's a logic/performance issue, don't burn LLM calls)
            if "timeout" in (last_error or "").lower():
                logger.warning(f"[{exec_id}] Timeout is not auto-repairable, giving up")
                break

            logger.info(f"[{exec_id}] Attempting LLM repair (round {repair_round + 1}/{max_repairs})")
            repair = self._attempt_repair(
                skill,
                last_error or "",
                last_traceback or "",
                context.params,
                contract,
            )
            repair_attempts_used += repair.attempts

            if not repair.success:
                logger.warning(f"[{exec_id}] Repair failed: {repair.error}")
                break

            current_skill_code = repair.repaired_code
            logger.info(f"[{exec_id}] Repair succeeded, retrying execution")

        # All attempts exhausted
        result = ExecutionResult(
            success=False,
            error=last_error or "Execution failed",
            exec_id=exec_id,
            latency_ms=total_latency_ms,
        )
        self._log_execution(exec_id, skill, context, result, start_time)
        return result

    def _run_with_retry(
        self,
        code: str,
        context_data: dict,
        contract: SkillContract,
        timeout_seconds: int,
    ) -> SandboxResult:
        """
        Run sandbox with retries for transient OS-level errors.
        Code errors (exceptions inside run()) are NOT retried here
        that's handled by the repair loop in execute()
        """
        import time

        last_result = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = run_in_sandbox(
                    skill_code=code,
                    context_data=context_data,
                    contract=contract,
                    timeout_seconds=timeout_seconds,
                )
                return result

            except SandboxTimeoutError as e:
                # Timeout is deterministic — don't retry
                from praxis.sandbox import SandboxResult
                return SandboxResult(
                    success=False,
                    error=str(e),
                    latency_ms=timeout_seconds * 1000,
                )

            except ContractViolationError as e:
                # Contract violation — don't retry
                from praxis.sandbox import SandboxResult
                return SandboxResult(success=False, error=str(e), latency_ms=0)

            except _TRANSIENT_SANDBOX_ERRORS as e:
                if attempt < self.config.max_retries:
                    logger.warning(
                        f"Transient sandbox error (attempt {attempt + 1}): {e}. Retrying..."
                    )
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    from praxis.sandbox import SandboxResult
                    return SandboxResult(
                        success=False,
                        error=f"Sandbox error after {attempt + 1} attempts: {e}",
                        latency_ms=0,
                    )

            except Exception as e:
                from praxis.sandbox import SandboxResult
                return SandboxResult(
                    success=False,
                    error=f"Unexpected sandbox error: {e}",
                    latency_ms=0,
                )

        # Should not reach here
        from praxis.sandbox import SandboxResult
        return SandboxResult(success=False, error="Max retries exceeded", latency_ms=0)

    def _attempt_repair(
        self,
        skill: Skill,
        error: str,
        traceback_str: str,
        inputs: dict,
        contract: SkillContract,
    ) -> RepairResult:
        """Delegate to refiner module."""
        return repair_skill(
            code=skill.code,
            error=error,
            traceback_str=traceback_str,
            inputs=inputs,
            llm_client=self.llm_client,
            contract=contract,
            max_attempts=self.config.max_repair_attempts,
        )

    def _apply_repair(self, skill: Skill, repaired_code: str) -> Skill:
        """Return a new Skill instance with repaired code + updated checksum."""
        import hashlib
        import copy
        repaired = copy.copy(skill)
        repaired.code = repaired_code
        repaired.checksum = hashlib.sha256(repaired_code.encode()).hexdigest()
        return repaired

    def _save_repaired_skill(self, skill: Skill, repaired_code: str) -> None:
        """Save repaired code as a new version in storage."""
        if not self.storage:
            return
        try:
            import copy
            repaired_skill = self._apply_repair(skill, repaired_code)
            repaired_skill.status = "draft"  # Repaired code starts as draft
            new_version_id = self.storage.save_skill(repaired_skill)
            logger.info(f"Saved repaired skill as version: {new_version_id}")
        except Exception as e:
            logger.error(f"Failed to save repaired skill: {e}")

    def _log_execution(
        self,
        exec_id: str,
        skill: Skill,
        context: ExecutionContext,
        result: ExecutionResult,
        start_time: datetime,
    ) -> None:
        """Write execution record to skill_executions table."""
        if not self.storage:
            return

        try:
            end_time = datetime.now()
            exit_status = "success" if result.success else "failure"

            self.storage.db.execute(
                """
                INSERT INTO skill_executions (
                    exec_id, version_id, start_time, end_time, exit_status,
                    failure_reason, latency_ms, llm_model, user_request
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exec_id,
                    skill.version_id,
                    start_time.isoformat(),
                    end_time.isoformat(),
                    exit_status,
                    result.error,
                    result.latency_ms,
                    context.llm_model,
                    context.user_request,
                ),
            )
            self.storage.db.commit()
        except Exception as e:
            # Logging failure should never crash execution
            logger.error(f"Failed to log execution {exec_id}: {e}")