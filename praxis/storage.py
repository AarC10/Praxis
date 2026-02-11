import json

from praxis.skill import Skill
from sqlite3 import Connection, Cursor, Row

class SkillStorage:
    db: Connection
    skills_dir: str

    def save_skill(self, skill: Skill) -> None:
        skill_id = self._generate_skill_id(skill.name)
        version = skill._get_next_version(skill_id)
        version_id = f"{skill_id}_v{version}"
        code_path = f"{self.skills_dir}/{version_id}.py"

        fpath = f"{self.skills_dir}/{version_id}.py"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, 'w') as f:
            f.write(skill.code)

        skill.version_id = version_id
        skill.version = version
        skill.code_path = code_path

        self.db.execute("""
            INSERT INTO skills (skill_id, name, short_desc, long_desc, version_id, 
                                version, code_path, contract_name, inputs_schema, 
                                outputs_schema, termination_condition, failure_modes, 
                                category, created_at, created_by, status, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            skill.skill_id,
            skill.name,
            skill.short_desc,
            skill.long_desc,
            skill.version_id,
            skill.version,
            skill.code_path,
            skill.contract_name,
            json.dumps(skill.inputs_schema),
            json.dumps(skill.outputs_schema),
            skill.termination_condition,
            json.dumps(skill.failure_modes),
            skill.category,
            skill.created_at.isoformat(),
            skill.created_by,
            skill.status,
            skill.checksum
        ))
        self.db.commit()

    def load_skill(self, name: str) -> Skill:
        pass

    def _generate_skill_id(self, name: str) -> str:
        return name.lower().replace(" ", "_")

    def _get_next_version(self, skill_id: str) -> int:
        cursor = self.db.execute("SELECT MAX(version) FROM skills WHERE skill_id = ?", (skill_id,))
        row = cursor.fetchone()
        return (row[0] or 0) + 1