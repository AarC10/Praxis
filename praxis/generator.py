import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from praxis.contracts import CONTRACTS, SkillContract
from praxis.llm_client import LLMClient, LLMResponse
from praxis.sandbox import validate_contract
from praxis.skill import Skill

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


@dataclass
class GenerationHints:
    input_keys: dict[str, str] = field(default_factory=dict)
    output_keys: dict[str, str] = field(default_factory=dict)
    category: Optional[str] = None
    created_by: str = "generator"


@dataclass
class GenerationResult:
    success: bool
    skill: Optional[Skill] = None
    version_id: Optional[str] = None
    was_reused: bool = False
    reuse_similarity: Optional[float] = None
    generation_attempts: int = 0
    error: Optional[str] = None
    schema_response: Optional[LLMResponse] = None
    code_response: Optional[LLMResponse] = None


@dataclass
class GeneratorConfig:
    reuse_similarity_threshold: float = 0.75
    max_generation_attempts: int = 3
    schema_temperature: float = 0.1
    code_temperature: float = 0.3
    max_tokens: int = 4096
    contract_name: str = "standalone_python"


def _build_contract_rules(contract: SkillContract) -> str:
    lines = [
        f"- Required function: {fn}(context) -> dict"
        for fn in contract.required_functions
    ]
    if contract.forbidden_imports:
        lines.append(f"- Forbidden imports: {', '.join(contract.forbidden_imports)}")
    if contract.forbidden_calls:
        lines.append(f"- Forbidden calls: {', '.join(f'{c}()' for c in contract.forbidden_calls)}")
    if contract.allowed_imports:
        lines.append(f"- Allowed standard imports: {', '.join(contract.allowed_imports)}")
    return "\n".join(lines)


def _extract_code_block(text: str) -> Optional[str]:
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _parse_schema_response(content: str) -> Optional[dict]:
    """Parse JSON schema from LLM response"""
    # Strip MD fences if present
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON object from surrounding text
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _search_existing_skills(
    request: str,
    llm_client: LLMClient,
    storage,
    threshold: float,
) -> Optional[tuple[Skill, float]]:
    """
    Embed the request and search for similar existing skills.
    Returns (skill, similarity) if a match above threshold is found, else None.
    Gracefully returns None if storage has no embeddings or sqlite-vec isn't available.
    """
    if storage is None:
        return None

    try:
        embedding_bytes = llm_client.embed(request)
    except Exception as e:
        logger.warning(f"Embedding failed, skipping skill reuse search: {e}")
        return None

    try:
        # sqlite-vec cosine search
        rows = storage.db.execute(
            """
            SELECT sv.version_id, (1 - vec_distance_cosine(sv.embedding, ?)) as similarity
            FROM skill_versions sv
            WHERE sv.embedding IS NOT NULL
              AND sv.status = 'active'
            ORDER BY similarity DESC
            LIMIT 1
            """,
            (embedding_bytes,),
        ).fetchall()

        if not rows:
            logger.debug("No embeddings in DB yet, will generate fresh")
            return None

        best = rows[0]
        similarity = best["similarity"]
        logger.info(f"Best existing skill match: {best['version_id']} (similarity={similarity:.3f})")

        if similarity >= threshold:
            skill = storage.load_skill(best["version_id"])
            return skill, similarity

        return None

    except Exception as e:
        logger.debug(f"Semantic search unavailable: {e}")
        return None


class SkillGenerator:
    def __init__(
        self,
        llm_client: LLMClient,
        storage=None,   # Optional[SkillStorage]
        config: Optional[GeneratorConfig] = None,
    ):
        self.llm_client = llm_client
        self.storage = storage
        self.config = config or GeneratorConfig()

    def generate(
        self,
        request: str,
        hints: Optional[GenerationHints] = None,
    ) -> GenerationResult:
        """
        Generate (or reuse) a skill for the given request.

        Args:
            request: Natural language description of what the skill should do
            hints: Optional rough schema hints — LLM fills in the rest

        Returns:
            GenerationResult
        """
        hints = hints or GenerationHints()
        contract = CONTRACTS.get(self.config.contract_name)
        if not contract:
            return GenerationResult(
                success=False,
                error=f"Unknown contract: {self.config.contract_name}",
            )

        # Semantic search for existing skill
        match = _search_existing_skills(
            request, self.llm_client, self.storage, self.config.reuse_similarity_threshold
        )
        if match:
            skill, similarity = match
            logger.info(f"Reusing existing skill '{skill.name}' (similarity={similarity:.3f})")
            return GenerationResult(
                success=True,
                skill=skill,
                version_id=skill.version_id,
                was_reused=True,
                reuse_similarity=similarity,
            )

        # Infer schema
        logger.info(f"Generating new skill for: {request!r}")
        schema_result = self._infer_schema(request, hints)
        if schema_result is None:
            return GenerationResult(
                success=False,
                error="Failed to infer skill schema from LLM response",
            )
        schema, schema_response = schema_result

        # Generate code (with retry on contract violations)
        contract_rules = _build_contract_rules(contract)
        code_response = None

        for attempt in range(1, self.config.max_generation_attempts + 1):
            logger.info(f"Code generation attempt {attempt}/{self.config.max_generation_attempts}")

            try:
                code_response = self.llm_client.generate(
                    _CODE_USER.format(
                        request=request,
                        name=schema.get("name", "unnamed_skill"),
                        desc=schema.get("long_desc", schema.get("short_desc", "")),
                        inputs_schema=json.dumps(schema.get("inputs_schema", {}), indent=2),
                        outputs_schema=json.dumps(schema.get("outputs_schema", {}), indent=2),
                        termination_condition=schema.get("termination_condition", "task complete"),
                        failure_modes=", ".join(schema.get("failure_modes", [])),
                        output_keys=", ".join(schema.get("outputs_schema", {}).keys()),
                    ),
                    system=_CODE_SYSTEM.format(contract_rules=contract_rules),
                    temperature=self.config.code_temperature,
                    max_tokens=self.config.max_tokens,
                )
            except Exception as e:
                logger.error(f"LLM code generation failed on attempt {attempt}: {e}")
                if attempt == self.config.max_generation_attempts:
                    return GenerationResult(
                        success=False,
                        error=f"LLM code generation failed: {e}",
                        schema_response=schema_response,
                        generation_attempts=attempt,
                    )
                continue

            code = _extract_code_block(code_response.content)
            if not code:
                logger.warning(f"Attempt {attempt}: LLM returned no code block")
                if attempt == self.config.max_generation_attempts:
                    return GenerationResult(
                        success=False,
                        error="LLM did not return a code block after all attempts",
                        schema_response=schema_response,
                        code_response=code_response,
                        generation_attempts=attempt,
                    )
                continue

            violations = validate_contract(code, contract)
            if violations:
                logger.warning(f"Attempt {attempt}: contract violations: {violations}")
                if attempt == self.config.max_generation_attempts:
                    return GenerationResult(
                        success=False,
                        error=f"Generated code has contract violations: {'; '.join(violations)}",
                        schema_response=schema_response,
                        code_response=code_response,
                        generation_attempts=attempt,
                    )
                # Feed violations back into the next attempt via modified request
                request = (
                    f"{request}\n\nPrevious attempt had these contract violations that MUST be fixed:\n"
                    + "\n".join(f"  - {v}" for v in violations)
                )
                continue

            # Code passed so build and save
            break
        else:
            # Loop exhausted without break
            return GenerationResult(
                success=False,
                error="Code generation exhausted all attempts",
                generation_attempts=self.config.max_generation_attempts,
            )

        skill = Skill(
            skill_id="",          # Set by storage on save
            name=schema.get("name", "unnamed_skill"),
            short_desc=schema.get("short_desc", ""),
            long_desc=schema.get("long_desc"),
            version_id="",        # Set by storage on save
            version=0,            # Set by storage on save
            code=code,
            code_path="",         # Set by storage on save
            checksum="",
            contract_name=self.config.contract_name,
            inputs_schema=schema.get("inputs_schema", {}),
            outputs_schema=schema.get("outputs_schema", {}),
            termination_condition=schema.get("termination_condition"),
            failure_modes=schema.get("failure_modes"),
            category=schema.get("category") or hints.category,
            created_by=hints.created_by,
            status="draft",
            generation_metadata={
                "schema_tokens_input": schema_response.tokens_input,
                "schema_tokens_output": schema_response.tokens_output,
                "code_tokens_input": code_response.tokens_input,
                "code_tokens_output": code_response.tokens_output,
                "generation_attempts": attempt,
                "llm_model": self.llm_client.get_model_name(),
            },
        )

        # Store embedding for future semantic search
        try:
            skill.embedding = self.llm_client.embed(skill.get_embedding_text())
        except Exception as e:
            logger.warning(f"Failed to generate skill embedding: {e}")

        version_id = None
        if self.storage:
            try:
                version_id = self.storage.save_skill(skill)
                logger.info(f"Saved skill as {version_id}")
            except Exception as e:
                logger.error(f"Failed to save skill: {e}")
                return GenerationResult(
                    success=False,
                    error=f"Skill generated but failed to save: {e}",
                    skill=skill,
                    schema_response=schema_response,
                    code_response=code_response,
                    generation_attempts=attempt,
                )
        else:
            logger.warning("No storage configured — skill not persisted")

        return GenerationResult(
            success=True,
            skill=skill,
            version_id=version_id,
            was_reused=False,
            generation_attempts=attempt,
            schema_response=schema_response,
            code_response=code_response,
        )

    def _infer_schema(
        self,
        request: str,
        hints: GenerationHints,
    ) -> Optional[tuple[dict, LLMResponse]]:
        """
        call 1: infer full schema from request + hints.
        Returns (schema_dict, llm_response) or None on failure.
        """
        user_prompt = _SCHEMA_USER.format(
            request=request,
            input_hints=json.dumps(hints.input_keys) if hints.input_keys else "none provided",
            output_hints=json.dumps(hints.output_keys) if hints.output_keys else "none provided",
        )

        try:
            response = self.llm_client.generate(
                user_prompt,
                system=_SCHEMA_SYSTEM,
                temperature=self.config.schema_temperature,
                max_tokens=1024,
            )
        except Exception as e:
            logger.error(f"Schema inference LLM call failed: {e}")
            return None

        schema = _parse_schema_response(response.content)
        if not schema:
            logger.error(f"Failed to parse schema from LLM response: {response.content[:200]}")
            return None

        required_keys = {"name", "inputs_schema", "outputs_schema"}
        missing = required_keys - schema.keys()
        if missing:
            logger.error(f"Schema missing required keys: {missing}")
            return None

        logger.info(
            f"Schema inferred: name={schema['name']!r}, "
            f"inputs={list(schema['inputs_schema'].keys())}, "
            f"outputs={list(schema['outputs_schema'].keys())}"
        )
        return schema, response