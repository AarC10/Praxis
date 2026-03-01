import sqlite3
import struct
from pathlib import Path

import pytest

from praxis.skill import Skill
from praxis.storage import SkillStorage

def _make_skill(**overrides) -> Skill:
    defaults = dict(
        skill_id="",
        name="add_numbers",
        short_desc="Adds two numbers",
        version_id="",
        version=0,
        code="def run(context):\n    return {'result': context.params['a'] + context.params['b']}",
        code_path="",
        checksum="",
        contract_name="standalone_python",
        inputs_schema={"a": "int", "b": "int"},
        outputs_schema={"result": "int"},
    )
    defaults.update(overrides)
    return Skill(**defaults)


def _fake_embedding(dim: int = 4) -> bytes:
    """Return a normalised float32 embedding as bytes."""
    val = 1.0 / dim ** 0.5
    return struct.pack(f"{dim}f", *([val] * dim))


@pytest.fixture()
def storage(tmp_path):
    schema_sql = (Path(__file__).parent.parent / "praxis" / "db" / "schema.sql").read_text()
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(schema_sql)
    return SkillStorage(db=db, data_dir=tmp_path)


class TestSaveAndLoad:
    def test_save_returns_version_id(self, storage):
        version_id = storage.save_skill(_make_skill())
        assert version_id == "add_numbers_v1"

    def test_load_by_version_id(self, storage):
        skill = _make_skill()
        version_id = storage.save_skill(skill)

        loaded = storage.load_skill(version_id)
        assert loaded.skill_id == "add_numbers"
        assert loaded.name == "add_numbers"
        assert loaded.version == 1
        assert loaded.code == skill.code
        assert loaded.inputs_schema == {"a": "int", "b": "int"}
        assert loaded.outputs_schema == {"result": "int"}

    def test_load_missing_raises(self, storage):
        with pytest.raises(ValueError, match="not found"):
            storage.load_skill("nonexistent_v99")

    def test_load_by_name_finds_active(self, storage):
        storage.save_skill(_make_skill())
        loaded = storage.load_skill_by_name("add_numbers")
        assert loaded.name == "add_numbers"
        assert loaded.status == "active"

    def test_load_by_name_missing_raises(self, storage):
        with pytest.raises(ValueError, match="No active version"):
            storage.load_skill_by_name("does_not_exist")


class TestAutoPromotion:
    def test_saved_skill_is_active(self, storage):
        version_id = storage.save_skill(_make_skill())
        loaded = storage.load_skill(version_id)
        assert loaded.status == "active"

    def test_skill_object_status_updated(self, storage):
        skill = _make_skill(status="draft")
        storage.save_skill(skill)
        assert skill.status == "active"

    def test_active_version_id_set_on_skills_table(self, storage):
        version_id = storage.save_skill(_make_skill())
        row = storage.db.execute(
            "SELECT active_version_id FROM skills WHERE skill_id = 'add_numbers'"
        ).fetchone()
        assert row["active_version_id"] == version_id

    def test_new_version_demotes_old(self, storage):
        v1 = storage.save_skill(_make_skill())
        v2 = storage.save_skill(_make_skill(code="def run(context): return {'result': 0}"))

        v1_row = storage.db.execute(
            "SELECT status FROM skill_versions WHERE version_id = ?", (v1,)
        ).fetchone()
        v2_row = storage.db.execute(
            "SELECT status FROM skill_versions WHERE version_id = ?", (v2,)
        ).fetchone()

        assert v1_row["status"] == "draft"
        assert v2_row["status"] == "active"

    def test_active_version_id_updated_on_new_version(self, storage):
        storage.save_skill(_make_skill())
        v2 = storage.save_skill(_make_skill(code="def run(context): return {'result': 0}"))

        row = storage.db.execute(
            "SELECT active_version_id FROM skills WHERE skill_id = 'add_numbers'"
        ).fetchone()
        assert row["active_version_id"] == v2

    def test_load_by_name_returns_latest_active(self, storage):
        storage.save_skill(_make_skill())
        storage.save_skill(_make_skill(code="def run(context): return {'result': 0}"))

        loaded = storage.load_skill_by_name("add_numbers")
        assert loaded.version == 2


class TestVersioning:
    def test_first_version_is_1(self, storage):
        skill = _make_skill()
        version_id = storage.save_skill(skill)
        assert skill.version == 1
        assert version_id == "add_numbers_v1"

    def test_second_save_increments_version(self, storage):
        storage.save_skill(_make_skill())
        v2_id = storage.save_skill(_make_skill(code="def run(context): return {'result': 0}"))
        assert v2_id == "add_numbers_v2"

    def test_different_skills_version_independently(self, storage):
        storage.save_skill(_make_skill(name="skill_a", short_desc="a"))
        storage.save_skill(_make_skill(name="skill_b", short_desc="b"))
        storage.save_skill(_make_skill(name="skill_a", short_desc="a",
                                       code="def run(context): return {}"))

        skill_a = storage.load_skill_by_name("skill_a")
        skill_b = storage.load_skill_by_name("skill_b")
        assert skill_a.version == 2
        assert skill_b.version == 1

    def test_load_by_name_with_explicit_version(self, storage):
        storage.save_skill(_make_skill())
        storage.save_skill(_make_skill(code="def run(context): return {'result': 0}"))

        loaded = storage.load_skill_by_name("add_numbers", version=1)
        assert loaded.version == 1


class TestCodePersistence:
    def test_code_written_to_file(self, storage, tmp_path):
        code = "def run(context):\n    return {'result': 99}"
        storage.save_skill(_make_skill(code=code))

        skill_file = tmp_path / "skills" / "add_numbers_v1.py"
        assert skill_file.exists()
        assert skill_file.read_text() == code

    def test_loaded_code_matches_saved(self, storage):
        code = "def run(context):\n    return {'result': context.params['a']}"
        version_id = storage.save_skill(_make_skill(code=code))

        loaded = storage.load_skill(version_id)
        assert loaded.code == code


class TestSchemaRoundTrip:
    def test_failure_modes_roundtrip(self, storage):
        skill = _make_skill(failure_modes=["division by zero", "type error"])
        version_id = storage.save_skill(skill)
        loaded = storage.load_skill(version_id)
        assert loaded.failure_modes == ["division by zero", "type error"]

    def test_none_failure_modes(self, storage):
        version_id = storage.save_skill(_make_skill(failure_modes=None))
        loaded = storage.load_skill(version_id)
        assert loaded.failure_modes is None

    def test_optional_fields_preserved(self, storage):
        skill = _make_skill(
            long_desc="A longer description",
            category="computation",
            termination_condition="when result is computed",
        )
        version_id = storage.save_skill(skill)
        loaded = storage.load_skill(version_id)
        assert loaded.long_desc == "A longer description"
        assert loaded.category == "computation"
        assert loaded.termination_condition == "when result is computed"

    def test_checksum_preserved(self, storage):
        skill = _make_skill()
        version_id = storage.save_skill(skill)
        loaded = storage.load_skill(version_id)
        assert loaded.checksum == skill.checksum
        assert loaded.verify_checksum()

    def test_embedding_roundtrip(self, storage):
        emb = _fake_embedding()
        skill = _make_skill(embedding=emb)
        version_id = storage.save_skill(skill)
        loaded = storage.load_skill(version_id)
        assert loaded.embedding == emb


class TestSemanticSearch:
    @pytest.fixture()
    def vec_storage(self, tmp_path):
        """Storage with sqlite-vec loaded; skip if unavailable."""
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not installed")

        schema_sql = (Path(__file__).parent.parent / "praxis" / "db" / "schema.sql").read_text()
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        db.executescript(schema_sql)
        return SkillStorage(db=db, data_dir=tmp_path)

    def test_vec_distance_cosine_finds_skill(self, vec_storage):
        """Saved skills with embeddings are reachable via vec_distance_cosine."""
        emb = _fake_embedding(dim=4)
        storage = vec_storage
        version_id = storage.save_skill(_make_skill(embedding=emb))

        rows = storage.db.execute(
            """
            SELECT sv.version_id,
                   (1 - vec_distance_cosine(sv.embedding, ?)) AS similarity
            FROM skill_versions sv
            WHERE sv.embedding IS NOT NULL
              AND sv.status = 'active'
            ORDER BY similarity DESC
            LIMIT 1
            """,
            (emb,),
        ).fetchall()

        assert len(rows) == 1
        assert rows[0]["version_id"] == version_id
        assert rows[0]["similarity"] == pytest.approx(1.0, abs=1e-5)

    def test_vec_search_ignores_draft_versions(self, vec_storage):
        """Demoted (draft) versions should not appear in the active search."""
        emb = _fake_embedding(dim=4)
        storage = vec_storage

        # Save v1 (auto-promoted), then v2 (auto-promoted → v1 demoted)
        storage.save_skill(_make_skill(embedding=emb))
        storage.save_skill(_make_skill(
            code="def run(context): return {'result': 0}",
            embedding=emb,
        ))

        rows = storage.db.execute(
            """
            SELECT sv.version_id
            FROM skill_versions sv
            WHERE sv.embedding IS NOT NULL
              AND sv.status = 'active'
            """,
        ).fetchall()

        assert len(rows) == 1
        assert rows[0]["version_id"] == "add_numbers_v2"
