import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Optional

from praxis.contracts import SkillContract

logger = logging.getLogger(__name__)

_RUNNER_TEMPLATE = textwrap.dedent("""
import json
import sys
import traceback

# --- injected skill code ---
{skill_code}
# --- end skill code ---

def _main():
    raw = sys.stdin.read()
    context_data = json.loads(raw)

    class ExecutionContext:
        def __init__(self, data):
            self.user_request = data.get("user_request", "")
            self.params = data.get("params", {{}})
            self.metadata = data.get("metadata", {{}})
            self.llm_model = data.get("llm_model")
            self.timeout_seconds = data.get("timeout_seconds", 30)

    ctx = ExecutionContext(context_data)

    try:
        result = run(ctx)
        if not isinstance(result, dict):
            raise TypeError(f"run() must return a dict, got {{type(result).__name__}}")
        print(json.dumps({{"ok": True, "data": result}}))
    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({{"ok": False, "error": str(e), "traceback": tb}}))
        sys.exit(1)

_main()
""")

_STRIPPED_ENV_KEYS = {
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE",
    "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX",
}


class SandboxError(Exception):
    """Raised when the sandbox itself fails (not the skill code)."""
    pass


class SandboxTimeoutError(SandboxError):
    """Raised when skill execution exceeds the timeout."""
    pass


class ContractViolationError(SandboxError):
    """Raised when skill code violates the contract before execution."""
    pass


class SandboxResult:
    def __init__(
        self,
        success: bool,
        data: Optional[dict] = None,
        error: Optional[str] = None,
        traceback: Optional[str] = None,
        stdout: str = "",
        stderr: str = "",
        latency_ms: int = 0,
    ):
        self.success = success
        self.data = data
        self.error = error
        self.traceback = traceback
        self.stdout = stdout
        self.stderr = stderr
        self.latency_ms = latency_ms


def validate_contract(code: str, contract: SkillContract) -> list[str]:
    """
    Perform static analysis of the skill code before running
    Should prevent forbidden imports and calls
    """
    violations = []

    for forbidden in contract.forbidden_imports:
        # Crude but effective for the common cases
        if f"import {forbidden}" in code or f"from {forbidden}" in code:
            violations.append(f"Forbidden import: {forbidden}")

    for forbidden_call in contract.forbidden_calls:
        # Check for call patterns - won't catch everything but catches obvious cases
        if f"{forbidden_call}(" in code:
            violations.append(f"Forbidden call: {forbidden_call}()")

    # Check required functions exist
    for required_fn in contract.required_functions:
        if f"def {required_fn}(" not in code:
            violations.append(f"Missing required function: {required_fn}")

    return violations


def run_in_sandbox(
    skill_code: str,
    context_data: dict[str, Any],
    contract: SkillContract,
    timeout_seconds: int = 30,
    python_executable: Optional[str] = None,
) -> SandboxResult:
    """
    Run skill code in an isolated subprocess.
    Args:
        skill_code: The skill's Python source
        context_data: Dict representation of ExecutionContext
        contract: The skill's contract for static validation
        timeout_seconds: Hard kill timeout
        python_executable: Path to Python interpreter (defaults to current)
    Returns:
        SandboxResult with success/failure info
    """
    import time

    # Static validation first — cheap, no subprocess needed
    violations = validate_contract(skill_code, contract)
    if violations:
        raise ContractViolationError(
            f"Contract violations found:\n" + "\n".join(f"  - {v}" for v in violations)
        )

    python = python_executable or sys.executable
    runner_code = _RUNNER_TEMPLATE.format(skill_code=skill_code)

    # Strip environment to avoid leaking credentials or affecting behavior
    clean_env = {k: v for k, v in os.environ.items() if k in _STRIPPED_ENV_KEYS}
    clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
    clean_env["PYTHONUNBUFFERED"] = "1"

    context_json = json.dumps(context_data)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="praxis_skill_",
        delete=False,
    ) as f:
        f.write(runner_code)
        tmp_path = f.name

    start = time.perf_counter()
    proc = None

    try:
        proc = subprocess.Popen(
            [python, tmp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=clean_env,
            # New process group so we can kill the whole tree
            start_new_session=True,
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(
                input=context_json.encode(),
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            # Kill the entire process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            raise SandboxTimeoutError(
                f"Skill exceeded timeout of {timeout_seconds}s"
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        stdout = stdout_bytes.decode(errors="replace").strip()
        stderr = stderr_bytes.decode(errors="replace").strip()

        if stderr:
            logger.debug(f"Skill stderr:\n{stderr}")

        # Last line of stdout should be our JSON result
        result_line = stdout.split("\n")[-1] if stdout else ""

        try:
            result_json = json.loads(result_line)
        except json.JSONDecodeError:
            return SandboxResult(
                success=False,
                error="Skill produced no valid JSON output",
                stdout=stdout,
                stderr=stderr,
                latency_ms=elapsed_ms,
            )

        if result_json.get("ok"):
            return SandboxResult(
                success=True,
                data=result_json["data"],
                stdout=stdout,
                stderr=stderr,
                latency_ms=elapsed_ms,
            )
        else:
            return SandboxResult(
                success=False,
                error=result_json.get("error", "Unknown error"),
                traceback=result_json.get("traceback"),
                stdout=stdout,
                stderr=stderr,
                latency_ms=elapsed_ms,
            )

    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass