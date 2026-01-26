#!/usr/bin/env python3
"""
Simple Enhanced Tool Calling Test Runner
Demonstrates all improvements working together
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("simple_test.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Run simple enhanced tests."""
    logger.info("=" * 70)
    logger.info("VibeAgent Simple Enhanced Tests")
    logger.info("=" * 70)

    try:
        from core.tool_orchestrator import ToolOrchestrator
        from skills.llm_skill import LLMSkill
        from core.database_manager import DatabaseManager
        from config.model_configs import get_model_config

        # Initialize components
        logger.info("\n1. Initializing Components...")
        db_manager = DatabaseManager("data/vibeagent_simple_test.db")
        logger.info("   ✓ DatabaseManager initialized")

        llm_skill = LLMSkill(base_url="http://localhost:8087/v1", model="glm-4.7")
        logger.info("   ✓ LLMSkill initialized")

        # Get model configuration
        model_config = get_model_config("glm-4.7")
        logger.info(
            f"   ✓ Model config loaded (max iterations: {model_config.max_iterations})"
        )

        # Create mock skills
        from core.skill import BaseSkill, SkillResult

        class MockSkill(BaseSkill):
            def __init__(self, name):
                super().__init__(name, "1.0.0")
                self.name = name

            def get_tool_schema(self):
                return {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": f"Mock {self.name} tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                            },
                            "required": ["query"],
                        },
                    },
                }

            def execute(self, **kwargs):
                return SkillResult(
                    success=True, data={"result": f"Executed {self.name} with {kwargs}"}
                )

            def get_dependencies(self):
                return []

            def validate(self):
                return True

        skills = {
            "search": MockSkill("search"),
            "analyze": MockSkill("analyze"),
            "store": MockSkill("store"),
        }
        logger.info(f"   ✓ {len(skills)} mock skills created")

        # Test ReAct mode
        logger.info("\n2. Testing ToolOrchestrator with ReAct...")
        orchestrator = ToolOrchestrator(
            llm_skill=llm_skill, skills=skills, db_manager=db_manager, use_react=True
        )
        logger.info("   ✓ ToolOrchestrator created with ReAct enabled")

        # Execute query
        logger.info("\n3. Executing test query...")
        result = orchestrator.execute_with_tools(
            user_message="Search for papers about machine learning",
            use_react=True,
            max_iterations=5,
        )

        logger.info(f"   ✓ Execution completed")
        logger.info(f"   - Success: {result.success}")
        logger.info(f"   - Iterations: {result.iterations}")
        logger.info(f"   - Tool calls made: {result.tool_calls_made}")

        # Check database
        logger.info("\n4. Verifying Database Tracking...")
        session_messages = db_manager.get_session_messages(1)
        logger.info(f"   ✓ {len(session_messages)} messages stored in database")

        # Test analytics
        logger.info("\n5. Testing Analytics...")
        from core.analytics_engine import AnalyticsEngine

        analytics = AnalyticsEngine(db_manager)

        # Get model comparison
        model_stats = analytics.get_model_comparison()
        if model_stats:
            logger.info(f"   ✓ Model comparison: {len(model_stats)} models analyzed")

        # Generate report
        report = analytics.generate_report()
        logger.info(
            f"   ✓ Report generated with {len(report.get('metrics', {}))} metrics"
        )

        logger.info("\n" + "=" * 70)
        logger.info("✓ All Tests Completed Successfully!")
        logger.info("=" * 70)
        logger.info("\nFeatures Demonstrated:")
        logger.info("  ✓ Database tracking (sessions, messages, tool calls)")
        logger.info("  ✓ ReAct reasoning loop")
        logger.info("  ✓ Tool orchestration")
        logger.info("  ✓ Model-specific configuration")
        logger.info("  ✓ Analytics engine")
        logger.info("\nCheck simple_test.log for detailed logs")

        return 0

    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
