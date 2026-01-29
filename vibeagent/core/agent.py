"""Core agent implementation."""

from datetime import datetime
from typing import Any

from .skill import BaseSkill, SkillResult, SkillStatus


class Agent:
    """Main agent that manages and executes skills."""

    def __init__(self, name: str = "VibeAgent"):
        self.name = name
        self.skills: dict[str, BaseSkill] = {}
        self.created_at = datetime.now().isoformat()
        self.memory: dict = {}

    def register_skill(self, skill: BaseSkill):
        """Register a new skill."""
        if skill.validate():
            self.skills[skill.name] = skill
            print(f"✅ Registered skill: {skill.name} v{skill.version}")
        else:
            print(f"❌ Failed to register skill: {skill.name} (validation failed)")

    def unregister_skill(self, skill_name: str):
        """Unregister a skill."""
        if skill_name in self.skills:
            del self.skills[skill_name]
            print(f"🗑️  Unregistered skill: {skill_name}")

    def get_skill(self, skill_name: str) -> BaseSkill | None:
        """Get a skill by name."""
        return self.skills.get(skill_name)

    def list_skills(self) -> list[dict]:
        """List all registered skills."""
        return [skill.get_info() for skill in self.skills.values()]

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Get all active skills as tool definitions for LLM use.

        Returns:
            List of tool definitions from active skills that implement get_tool_schema().
            Each tool definition contains the schema returned by the skill's get_tool_schema() method.
        """
        tools = []
        for skill in self.skills.values():
            if skill.status == SkillStatus.ACTIVE:
                if hasattr(skill, "get_tool_schema") and callable(skill.get_tool_schema):
                    try:
                        schema = skill.get_tool_schema()
                        if schema:
                            tools.append(schema)
                    except Exception:
                        pass
        return tools

    def execute_skill(self, skill_name: str, **kwargs) -> SkillResult:
        """Execute a skill by name."""
        skill = self.get_skill(skill_name)
        if not skill:
            return SkillResult(success=False, error=f"Skill '{skill_name}' not found")

        if skill.status != SkillStatus.ACTIVE:
            return SkillResult(
                success=False,
                error=f"Skill '{skill_name}' is not active (status: {skill.status.value})",
            )

        try:
            result = skill.execute(**kwargs)
            skill._record_usage()
            if result.success:
                skill.status = SkillStatus.ACTIVE
            else:
                skill._record_error()
            return result
        except Exception as e:
            skill._record_error()
            return SkillResult(success=False, error=f"Skill execution failed: {str(e)}")

    def health_check(self) -> dict[str, bool]:
        """Check health of all skills."""
        return {skill_name: skill.health_check() for skill_name, skill in self.skills.items()}

    def self_heal(self):
        """Attempt to heal broken skills."""
        print("🔧 Self-healing check...")
        for skill_name, skill in self.skills.items():
            if skill.status == SkillStatus.ERROR:
                print(f"   Attempting to heal: {skill_name}")
                if skill.health_check():
                    skill.activate()
                    print(f"   ✅ Healed: {skill_name}")
                else:
                    print(f"   ❌ Failed to heal: {skill_name}")

    def update_skills(self):
        """Check for and apply updates to skills."""
        print("🔄 Checking for skill updates...")
        for skill_name, skill in self.skills.items():
            skill.status = SkillStatus.UPDATING
            # In a real implementation, this would check for updates
            # For now, we just mark it as active again
            skill.activate()
            print(f"   ✅ Updated: {skill_name}")

    def remember(self, key: str, value: Any):
        """Store information in agent memory."""
        self.memory[key] = {"value": value, "timestamp": datetime.now().isoformat()}

    def recall(self, key: str) -> Any | None:
        """Retrieve information from agent memory."""
        if key in self.memory:
            return self.memory[key]["value"]
        return None

    def get_status(self) -> dict:
        """Get agent status."""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "skills_count": len(self.skills),
            "skills": self.list_skills(),
            "health": self.health_check(),
        }
