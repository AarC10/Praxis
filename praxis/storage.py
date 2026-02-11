from praxis.skill import Skill


class SkillStorage:
    def save(self, skill: Skill) -> None:
        raise NotImplementedError

    def load(self, name: str) -> Skill:
        raise NotImplementedError