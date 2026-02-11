import pytest
from praxis.skill import Skill, SkillMatch, ExecutionResult
from datetime import datetime

def test_skill_creation():
    skill = Skill(
        skill_id="test_skill",
        name="Test Skill",
        short_desc="A test skill",
        version_id="test_skill_v1",
        version=1,
        code="def run(context): return {'result': 42}",
        code_path="skills/test_skill_v1.py",
        checksum="",
        contract_name="standalone_python",
        inputs_schema={"x": "int"},
        outputs_schema={"result": "int"}
    )

    assert skill.skill_id == "test_skill"
    assert skill.checksum != ""
    assert skill.status == "draft"


def test_checksum_verification():
    skill = Skill(
        skill_id="test",
        name="Test",
        short_desc="Test",
        version_id="test_v1",
        version=1,
        code="def run(): pass",
        code_path="skills/test.py",
        checksum="",
        contract_name="standalone_python",
        inputs_schema={},
        outputs_schema={}
    )

    original_checksum = skill.checksum
    assert skill.verify_checksum()

    skill.code = "def run(): malicious()"
    assert not skill.verify_checksum()

    skill.code = "def run(): pass"
    assert skill.verify_checksum()


def test_input_validation():
    skill = Skill(
        skill_id="test",
        name="Test",
        short_desc="Test",
        version_id="test_v1",
        version=1,
        code="def run(context): pass",
        code_path="skills/test.py",
        checksum="",
        contract_name="standalone_python",
        inputs_schema={"x": "int", "y": "float"},
        outputs_schema={}
    )

    # Valid
    is_valid, errors = skill.matches_input_schema({"x": 1, "y": 2.0})
    assert is_valid
    assert len(errors) == 0

    # Missing param
    is_valid, errors = skill.matches_input_schema({"x": 1})
    assert not is_valid
    assert "y" in errors[0]


def test_embedding_text_generation():
    skill = Skill(
        skill_id="test",
        name="Detect Objects",
        short_desc="YOLO-based detection",
        version_id="test_v1",
        version=1,
        code="",
        code_path="skills/test.py",
        checksum="",
        contract_name="standalone_python",
        inputs_schema={"image": "ndarray"},
        outputs_schema={"boxes": "list"},
        category="perception",
        termination_condition="image_processed"
    )

    text = skill.get_embedding_text()
    assert "Detect Objects" in text
    assert "YOLO" in text
    assert "perception" in text
    assert "image_processed" in text


def test_skill_serialization():
    skill = Skill(
        skill_id="test",
        name="Test",
        short_desc="Test",
        version_id="test_v1",
        version=1,
        code="def run(): pass",
        code_path="skills/test.py",
        checksum="abc123",
        contract_name="standalone_python",
        inputs_schema={"x": "int"},
        outputs_schema={"y": "int"}
    )

    # Serialize
    data = skill.to_dict()
    assert data['skill_id'] == "test"
    assert data['checksum'] == "abc123"

    # Deserialize
    skill2 = Skill.from_dict(data, code="def run(): pass")
    assert skill2.skill_id == skill.skill_id
    assert skill2.checksum == skill.checksum


def test_execution_result():
    result = ExecutionResult(
        success=True,
        data={"output": 42},
        exec_id="exec_123",
        latency_ms=150
    )

    assert result.success
    assert result.data["output"] == 42

    result_dict = result.to_dict()
    assert result_dict['success'] == True
    assert result_dict['latency_ms'] == 150