import logging
import re
import json

from dataclasses import dataclass
from typing import Optional

from praxis.contracts import SkillContract, CONTRACTS
from praxis.llm_client import LLMClient, LLMResponse
from praxis.sandbox import validate_contract

logger = logging.getLogger(__name__)

_REPAIR_SYSTEM_PROMPT = """You are an expert Python debugger. Your job is to fix broken skill code.

A "skill" is a Python module with a required `run(context)` function that returns a dict.
The context object has:
  - context.params: dict of input parameters
  - context.user_request: str describing what the user wants
  - context.metadata: dict of additional metadata

Rules you MUST follow:
{contract_rules}

Return ONLY the corrected Python code inside a ```python ... ``` block.
No explanation before or after. Just the fixed code block.
"""

_REPAIR_USER_PROMPT = """The following skill code failed during execution.

## Original Code
```python
{code}
```

## Error
```
{error}
```

## Traceback
```
{traceback}
```

## Inputs the skill received
```json
{inputs}
```

Fix the code so it handles the inputs correctly and returns a valid dict from `run(context)`.
"""


@dataclass
class RepairResult:
    success: bool
    repaired_code: Optional[str] = None
    attempts: int = 0
    error: Optional[str] = None
    llm_response: Optional[LLMResponse] = None


def _build_contract_rules(contract: SkillContract) -> str:
    lines = [
        f"- Required function: {fn}(context) -> dict" for fn in contract.required_functions
    ]
    if contract.forbidden_imports:
        lines.append(f"- Forbidden imports: {', '.join(contract.forbidden_imports)}")
    if contract.forbidden_calls:
        lines.append(f"- Forbidden calls: {', '.join(f'{c}()' for c in contract.forbidden_calls)}")
    if contract.allowed_imports:
        lines.append(f"- Allowed imports: {', '.join(contract.allowed_imports)}")
    lines.append(f"- Max lines: {contract.max_lines}")
    return "\n".join(lines)


def _extract_code_block(llm_output: str) -> Optional[str]:
    """Extract Python code from a MD code block."""
    match = re.search(r"```python\s*\n(.*?)```", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)```", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def repair_skill(
    code: str,
    error: str,
    traceback_str: str,
    inputs: dict,
    llm_client: LLMClient,
    contract: Optional[SkillContract] = None,
    max_attempts: int = 3,
) -> RepairResult:
    """
    Attempt to repair failing skill code

    Args:
        code: The broken skill code
        error: The error message from exec
        traceback_str: Full traceback string
        inputs: The input params that caused the failure
        llm_client: LLM client to use for repair
        contract: Skill contract for validation (defaults to standalone_python)
        max_attempts: Max repair iterations

    Returns:
        RepairResult indicating whether repair succeeded and the fixed code
    """
    if contract is None:
        contract = CONTRACTS["standalone_python"]

    contract_rules = _build_contract_rules(contract)
    system_prompt = _REPAIR_SYSTEM_PROMPT.format(contract_rules=contract_rules)

    current_code = code
    current_error = error
    current_traceback = traceback_str

    for attempt in range(1, max_attempts + 1):
        logger.info(f"Repair attempt {attempt}/{max_attempts}")

        user_prompt = _REPAIR_USER_PROMPT.format(
            code=current_code,
            error=current_error,
            traceback=current_traceback or "No traceback available",
            inputs=json.dumps(inputs, indent=2, default=str),
        )

        try:
            llm_response = llm_client.generate(
                user_prompt,
                system=system_prompt,
                temperature=0.2,  # Low temp for deterministic fixes
                max_tokens=4096,
            )
        except Exception as e:
            logger.error(f"LLM call failed during repair attempt {attempt}: {e}")
            return RepairResult(
                success=False,
                attempts=attempt,
                error=f"LLM call failed: {e}",
            )

        repaired_code = _extract_code_block(llm_response.content)
        if not repaired_code:
            logger.warning(f"Attempt {attempt}: LLM returned no code block")
            current_error = "LLM did not return a code block"
            current_traceback = ""
            current_code = current_code  # Keep existing code
            continue

        # Validate against contract before accepting
        violations = validate_contract(repaired_code, contract)
        if violations:
            logger.warning(
                f"Attempt {attempt}: Repaired code has contract violations: {violations}"
            )
            current_error = f"Contract violations in repaired code: {'; '.join(violations)}"
            current_traceback = ""
            current_code = repaired_code  # Feed it back
            continue

        logger.info(f"Repair succeeded on attempt {attempt}")
        return RepairResult(
            success=True,
            repaired_code=repaired_code,
            attempts=attempt,
            llm_response=llm_response,
        )

    return RepairResult(
        success=False,
        attempts=max_attempts,
        error=f"Failed to repair after {max_attempts} attempts. Last error: {current_error}",
    )