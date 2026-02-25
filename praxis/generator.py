import logging


logger = logging.getLogger(__name__)

_SCHEMA_SYSTEM = """You are a precise API designer. Given a task description, output a JSON schema definition for a Python skill.

A skill has:
- A `run(context)` function where context.params contains the inputs
- Returns a dict matching the outputs schema

Respond with ONLY a JSON object — no markdown, no explanation:
{
  "name": "short_snake_case_name",
  "short_desc": "One sentence description",
  "long_desc": "2-3 sentence description of what this skill does and how",
  "category": "one of: perception, manipulation, planning, navigation, computation, io, util",
  "inputs_schema": {"param_name": "type", ...},
  "outputs_schema": {"output_name": "type", ...},
  "termination_condition": "brief description of when this skill is done",
  "failure_modes": ["list", "of", "ways", "this", "could", "fail"]
}

Types should be simple strings: int, float, str, bool, list, dict, ndarray, bytes, any.
Keep schemas minimal — only what's strictly needed.
"""

_SCHEMA_USER = """Task: {request}

Known inputs (hints, may be incomplete): {input_hints}
Known outputs (hints, may be incomplete): {output_hints}

Infer the complete schema. Add any inputs or outputs that are obviously needed even if not listed.
"""

_CODE_SYSTEM = """You are an expert Python developer writing robot skills.

A skill is a Python module with a single `run(context)` function:
- context.params: dict — input parameters (types defined by schema)
- context.user_request: str — the original user request
- context.metadata: dict — additional runtime metadata
- Returns: dict — must match the outputs schema exactly

Contract rules you MUST follow:
{contract_rules}

Write clean, well-commented, production-quality code.
Handle edge cases. Include a docstring on run().
Return ONLY the Python code inside a ```python ... ``` block.
"""

_CODE_USER = """Write a skill that does the following:

Task: {request}

Skill name: {name}
Description: {desc}

Inputs schema: {inputs_schema}
Outputs schema: {outputs_schema}

Termination condition: {termination_condition}
Potential failure modes: {failure_modes}

The run(context) function must return a dict with exactly these keys: {output_keys}
"""

