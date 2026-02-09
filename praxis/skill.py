from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json
import hashlib


@dataclass
class Skill:
    # Identity
    skill_id: str
    name: str
    short_desc: str
    version_id: str
    version: int

    # Code
    code: str
    code_path: str
    checksum: str

    # Contract
    contract_name: str
    inputs_schema: Dict[str, Any]
    outputs_schema: Dict[str, Any]
    termination_condition: Optional[str] = None
    failure_modes: Optional[List[str]] = None

    # Metadata
    long_desc: Optional[str] = None
    category: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "unknown"
    status: str = "draft"

    # Embeddings
    embedding: Optional[bytes] = None

    # Generation metadata
    generation_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate and compute derived fields"""
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        return hashlib.sha256(self.code.encode()).hexdigest()

    def verify_checksum(self) -> bool:
        return self._compute_checksum() == self.checksum

    def to_dict(self) -> Dict[str, Any]:
        return {
            'skill_id': self.skill_id,
            'name': self.name,
            'short_desc': self.short_desc,
            'long_desc': self.long_desc,
            'version_id': self.version_id,
            'version': self.version,
            'code_path': self.code_path,
            'contract_name': self.contract_name,
            'inputs_schema': self.inputs_schema,
            'outputs_schema': self.outputs_schema,
            'termination_condition': self.termination_condition,
            'failure_modes': self.failure_modes,
            'category': self.category,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'status': self.status,
            'checksum': self.checksum,
            'generation_metadata': self.generation_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], code: str) -> 'Skill':
        data = data.copy()
        data['code'] = code
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

    def get_embedding_text(self) -> str:
        """
        Generate text for embedding that includes all searchable content.
        * Use for semantic searhc
        """
        parts = [
            self.name,
            self.short_desc,
            self.long_desc or "",
            self.category or "",
            f"inputs: {json.dumps(self.inputs_schema)}",
            f"outputs: {json.dumps(self.outputs_schema)}",
        ]

        if self.termination_condition:
            parts.append(f"terminates when: {self.termination_condition}")

        if self.failure_modes:
            parts.append(f"failure modes: {', '.join(self.failure_modes)}")

        return " ".join(parts)

    def matches_input_schema(self, params: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Check if provided parameters match input schema.
        Returns (is_valid, list_of_errors)
        """
        errors = []

        # Check required inputs are present
        for param_name, param_type in self.inputs_schema.items():
            if param_name not in params:
                errors.append(f"Missing required parameter: {param_name}")
            # TODO: Type checks

        return len(errors) == 0, errors

    def validates_output(self, result: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Check if execution result matches output schema.
        Returns (is_valid, list_of_errors)
        """
        errors = []

        for output_name in self.outputs_schema.keys():
            if output_name not in result:
                errors.append(f"Missing expected output: {output_name}")
            # TODO: Add type checks :)))))

        return len(errors) == 0, errors


@dataclass
class SkillMatch:
    """Result from semantic search"""
    skill: Skill
    distance: float  # Embedding distance (lower is more similar)

    @property
    def similarity(self) -> float:
        """Convert distance to similarity score [0-1]"""
        # assume cosine distance, convert to similarity
        return 1.0 - self.distance

    def __lt__(self, other: 'SkillMatch') -> bool:
        """For sorting by similarity (higher is better)"""
        return self.distance < other.distance


@dataclass
class ExecutionResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    exec_id: Optional[str] = None
    latency_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'exec_id': self.exec_id,
            'latency_ms': self.latency_ms
        }


@dataclass
class ExecutionContext:
    user_request: str
    params: Dict[str, Any]
    llm_model: Optional[str] = None
    timeout_seconds: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)