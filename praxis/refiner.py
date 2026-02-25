import logging

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

