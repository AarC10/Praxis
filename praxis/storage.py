import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from sqlite3 import Connection

from praxis.skill import Skill


class SkillStorage:
    def __init__(self, db: Connection, data_dir: Path):
        self.db = db
        self.data_dir = data_dir
        self.skills_dir = data_dir / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def save_skill(self, skill: Skill) -> str:
        skill_id = self._generate_skill_id(skill.name)
        version = self._get_next_version(skill_id)
        version_id = f"{skill_id}_v{version}"
        code_path = f"skills/{version_id}.py"

        fpath = self.data_dir / code_path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(skill.code)

        skill.skill_id = skill_id
        skill.version_id = version_id
        skill.version = version
        skill.code_path = code_path

        # TODO: Shrink to one db?
        self.db.execute("""
                        INSERT
                        OR IGNORE INTO skills (skill_id, name, short_desc, long_desc, 
                                          category, created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            skill_id,
                            skill.name,
                            skill.short_desc,
                            skill.long_desc,
                            skill.category,
                            skill.created_at.isoformat(),
                            skill.created_by
                        ))

        self.db.execute("""
                        INSERT INTO skill_versions (version_id, skill_id, version, status,
                                                    created_at, code_path, contract_name, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            version_id,
                            skill_id,
                            version,
                            skill.status,
                            skill.created_at.isoformat(),
                            code_path,
                            skill.contract_name,
                            skill.checksum
                        ))

        interface_id = f"{version_id}_interface"
        self.db.execute("""
                        INSERT INTO skill_interfaces (interface_id, version_id, inputs_schema,
                                                      outputs_schema, termination_condition, failure_modes)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            interface_id,
                            version_id,
                            json.dumps(skill.inputs_schema),
                            json.dumps(skill.outputs_schema),
                            skill.termination_condition,
                            json.dumps(skill.failure_modes) if skill.failure_modes else None
                        ))

        self.db.commit()
        return version_id

    def load_skill(self, version_id: str) -> Skill:
        row = self.db.execute("""
                              SELECT s.skill_id,
                                     s.name,
                                     s.short_desc,
                                     s.long_desc,
                                     s.category,
                                     s.created_by,
                                     sv.version_id,
                                     sv.version,
                                     sv.code_path,
                                     sv.contract_name,
                                     sv.checksum,
                                     sv.status,
                                     sv.created_at,
                                     si.inputs_schema,
                                     si.outputs_schema,
                                     si.termination_condition,
                                     si.failure_modes
                              FROM skill_versions sv
                                       JOIN skills s ON sv.skill_id = s.skill_id
                                       JOIN skill_interfaces si ON sv.version_id = si.version_id
                              WHERE sv.version_id = ?
                              """, (version_id,)).fetchone()

        if not row:
            raise ValueError(f"Skill version not found: {version_id}")

        code_path = self.data_dir / row['code_path']
        code = code_path.read_text()

        # TODO: Has to be a better way to do this
        return Skill(
            skill_id=row['skill_id'],
            name=row['name'],
            short_desc=row['short_desc'],
            long_desc=row['long_desc'],
            version_id=row['version_id'],
            version=row['version'],
            code=code,
            code_path=row['code_path'],
            checksum=row['checksum'],
            contract_name=row['contract_name'],
            inputs_schema=json.loads(row['inputs_schema']),
            outputs_schema=json.loads(row['outputs_schema']),
            termination_condition=row['termination_condition'],
            failure_modes=json.loads(row['failure_modes']) if row['failure_modes'] else None,
            category=row['category'],
            created_at=datetime.fromisoformat(row['created_at']),
            created_by=row['created_by'],
            status=row['status']
        )

    def load_skill_by_name(self, name: str, version: Optional[int] = None) -> Skill:
        if version:
            version_id = f"{self._generate_skill_id(name)}_v{version}"
            return self.load_skill(version_id)

        row = self.db.execute("""
                              SELECT sv.version_id
                              FROM skill_versions sv
                                       JOIN skills s ON sv.skill_id = s.skill_id
                              WHERE s.name = ?
                                AND sv.status = 'active'
                              ORDER BY sv.version DESC LIMIT 1
                              """, (name,)).fetchone()

        if not row:
            raise ValueError(f"No active version found for skill: {name}")

        return self.load_skill(row['version_id'])

    def _generate_skill_id(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")

    def _get_next_version(self, skill_id: str) -> int:
        result = self.db.execute("""
                                 SELECT MAX(version) as max_version
                                 FROM skill_versions
                                 WHERE skill_id = ?
                                 """, (skill_id,)).fetchone()

        if result['max_version'] is None:
            return 1
        return result['max_version'] + 1