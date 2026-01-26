"""Example usage of Plan-and-Execute orchestrator."""

import logging
from skills.llm_skill import LLMSkill
from core.plan_execute_orchestrator import (
    PlanExecuteOrchestrator,
    PlanExecuteOrchestratorConfig,
)
from core.database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate Plan-and-Execute orchestrator."""
    llm_skill = LLMSkill(
        base_url="http://localhost:11434/v1",
        model="llama3.2",
    )

    db_manager = DatabaseManager("data/vibeagent.db")

    config = PlanExecuteOrchestratorConfig(
        max_plan_steps=15,
        max_plan_depth=5,
        plan_validation_strictness="moderate",
        adaptation_sensitivity=0.7,
        enable_parallel=True,
        max_parallel_steps=3,
        enable_plan_learning=True,
        fallback_to_sequential=True,
    )

    orchestrator = PlanExecuteOrchestrator(
        llm_skill=llm_skill,
        skills={},
        db_manager=db_manager,
        config=config,
    )

    user_message = "Research the latest developments in quantum computing and summarize the key findings"

    logger.info(f"Executing task with Plan-and-Execute: {user_message}")

    result = orchestrator.execute_with_tools(user_message, max_iterations=5)

    if result.success:
        logger.info(f"Task completed successfully!")
        logger.info(f"Response: {result.final_response}")
        logger.info(f"Iterations: {result.iterations}")
        logger.info(f"Tool calls made: {result.tool_calls_made}")
    else:
        logger.error(f"Task failed: {result.error}")


if __name__ == "__main__":
    main()
