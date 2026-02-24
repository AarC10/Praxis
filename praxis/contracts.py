from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SkillContract:
    name: str
    language: str
    runtime: str
    required_functions: List[str]
    function_signatures: Dict[str, Dict]
    allowed_imports: List[str]
    forbidden_imports: List[str]
    forbidden_calls: List[str]
    max_lines: int = 500
    timeout_seconds: int = 30

STANDALONE_PYTHON = SkillContract(
    name="standalone_python",
    language="python",
    runtime="standalone",
    required_functions=["run"],
    function_signatures={
        "run": {
            "params": ["context"],
            "returns": "dict"
        }
    },
    allowed_imports=[
        "numpy", "cv2", "PIL", "matplotlib",
        "scipy", "sklearn", "math", "json",
        "time", "datetime", "collections"
    ],
    forbidden_imports=[
        "os", "sys", "subprocess", "importlib",
        "eval", "exec", "compile", "__import__"
    ],
    forbidden_calls=[
        "eval", "exec", "compile", "open",
        "input", "raw_input"
    ]
)

CONTRACTS = {
    "standalone_python": STANDALONE_PYTHON,
}