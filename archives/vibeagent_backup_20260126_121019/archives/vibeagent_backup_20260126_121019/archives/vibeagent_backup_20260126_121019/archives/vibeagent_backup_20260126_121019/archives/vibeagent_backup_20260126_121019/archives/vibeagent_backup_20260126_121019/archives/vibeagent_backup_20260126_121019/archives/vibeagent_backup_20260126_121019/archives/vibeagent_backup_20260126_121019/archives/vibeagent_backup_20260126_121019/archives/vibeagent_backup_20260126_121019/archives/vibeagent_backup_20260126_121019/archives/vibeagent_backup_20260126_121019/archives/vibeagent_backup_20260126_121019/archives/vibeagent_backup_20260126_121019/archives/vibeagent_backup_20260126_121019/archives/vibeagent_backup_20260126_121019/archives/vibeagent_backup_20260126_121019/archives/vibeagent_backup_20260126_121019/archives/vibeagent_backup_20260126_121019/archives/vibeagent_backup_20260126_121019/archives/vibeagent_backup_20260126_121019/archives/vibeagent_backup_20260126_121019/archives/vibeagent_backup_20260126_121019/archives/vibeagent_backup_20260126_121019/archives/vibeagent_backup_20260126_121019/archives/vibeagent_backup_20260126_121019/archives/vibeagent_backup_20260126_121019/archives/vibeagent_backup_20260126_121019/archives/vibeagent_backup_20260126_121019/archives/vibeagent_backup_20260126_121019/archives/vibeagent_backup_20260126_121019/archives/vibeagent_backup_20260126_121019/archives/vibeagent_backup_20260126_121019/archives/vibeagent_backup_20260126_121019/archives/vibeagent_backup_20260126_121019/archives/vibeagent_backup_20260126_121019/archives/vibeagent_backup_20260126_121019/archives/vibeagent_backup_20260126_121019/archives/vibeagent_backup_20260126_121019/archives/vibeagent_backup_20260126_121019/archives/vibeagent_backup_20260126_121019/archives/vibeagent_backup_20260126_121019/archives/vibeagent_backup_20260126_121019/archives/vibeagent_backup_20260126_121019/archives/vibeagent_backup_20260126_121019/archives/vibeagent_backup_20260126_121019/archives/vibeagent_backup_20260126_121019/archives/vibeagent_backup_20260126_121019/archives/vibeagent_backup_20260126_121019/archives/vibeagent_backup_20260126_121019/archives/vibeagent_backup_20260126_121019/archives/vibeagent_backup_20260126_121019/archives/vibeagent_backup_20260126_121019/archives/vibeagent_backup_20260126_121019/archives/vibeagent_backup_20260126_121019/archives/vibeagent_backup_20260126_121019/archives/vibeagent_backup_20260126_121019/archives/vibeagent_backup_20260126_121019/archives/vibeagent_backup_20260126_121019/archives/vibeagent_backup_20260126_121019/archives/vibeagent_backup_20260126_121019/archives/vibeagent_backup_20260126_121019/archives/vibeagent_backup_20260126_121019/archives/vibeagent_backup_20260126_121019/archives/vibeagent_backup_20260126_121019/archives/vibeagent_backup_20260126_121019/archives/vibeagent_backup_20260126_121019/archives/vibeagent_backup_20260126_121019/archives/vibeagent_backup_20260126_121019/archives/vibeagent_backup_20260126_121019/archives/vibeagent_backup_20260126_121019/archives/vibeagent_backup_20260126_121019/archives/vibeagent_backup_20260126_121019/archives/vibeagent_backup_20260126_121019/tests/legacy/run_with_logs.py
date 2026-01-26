#!/usr/bin/env python3
"""
Run VibeAgent with full logging and all features enabled.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("vibeagent.log")],
)

# Set specific log levels for different modules
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)

# Import after logging configuration
from core import Agent
from core.database_manager import DatabaseManager
from core.tool_orchestrator import ToolOrchestrator
from core.plan_execute_orchestrator import PlanExecuteOrchestrator
from core.tot_orchestrator import ToTOrchestrator
from skills import ArxivSkill, ScraperSkill, LLMSkill, PocketBaseSkill
from config import Config

logger = logging.getLogger(__name__)


def setup_database():
    """Setup database with full schema."""
    db_path = Path("data/vibeagent.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db_manager = DatabaseManager(db_path=str(db_path))
    logger.info(f"✅ Database initialized at {db_path}")

    # Verify all tables exist
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"📊 Database tables: {len(tables)} tables")
        for table in tables:
            logger.info(f"   - {table}")

    return db_manager


def main():
    """Main entry point with full logging."""
    logger.info("=" * 80)
    logger.info("🤖 VibeAgent Starting with Full Logging")
    logger.info("=" * 80)

    # Load configuration
    config = Config()
    logger.info(f"📝 Configuration loaded from {config.config_path}")
    logger.info(f"   Agent Name: {config.get('agent', 'name')}")
    logger.info(f"   Version: {config.get('agent', 'version')}")

    # Setup database
    logger.info("\n" + "=" * 80)
    logger.info("🗄️  Database Setup")
    logger.info("=" * 80)
    db_manager = setup_database()

    # Initialize agent
    logger.info("\n" + "=" * 80)
    logger.info("🤖 Agent Initialization")
    logger.info("=" * 80)
    agent = Agent(name=config.get("agent", "name"))
    logger.info(f"✅ Agent '{agent.name}' initialized")

    # Register skills
    logger.info("\n" + "=" * 80)
    logger.info("🔧 Registering Skills")
    logger.info("=" * 80)

    skills_count = 0

    # ArXiv Skill
    if config.get("skills", "arxiv", "enabled"):
        logger.info("📚 Registering ArXiv Skill...")
        arxiv_skill = ArxivSkill()
        agent.register_skill(arxiv_skill)
        skills_count += 1
        logger.info(f"   ✅ ArXiv Skill registered: {arxiv_skill.name}")

    # Scraper Skill
    if config.get("skills", "scraper", "enabled"):
        logger.info("🌐 Registering Scraper Skill...")
        scraper_skill = ScraperSkill()
        agent.register_skill(scraper_skill)
        skills_count += 1
        logger.info(f"   ✅ Scraper Skill registered: {scraper_skill.name}")

    # LLM Skill
    llm_skill = None
    if config.get("skills", "llm", "enabled"):
        logger.info("🧠 Registering LLM Skill...")
        llm_skill = LLMSkill(
            base_url=config.get("skills", "llm", "base_url"),
            model=config.get("skills", "llm", "model"),
        )
        agent.register_skill(llm_skill)
        skills_count += 1
        logger.info(f"   ✅ LLM Skill registered: {llm_skill.name}")
        logger.info(f"      Base URL: {llm_skill.base_url}")
        logger.info(f"      Model: {llm_skill.model}")

    # PocketBase Skill
    if config.get("pocketbase", "url"):
        logger.info("💾 Registering PocketBase Skill...")
        pocketbase_skill = PocketBaseSkill(
            base_url=config.get("pocketbase", "url"),
            email=config.get("pocketbase", "email"),
            password=config.get("pocketbase", "password"),
        )
        agent.register_skill(pocketbase_skill)
        skills_count += 1
        logger.info(f"   ✅ PocketBase Skill registered: {pocketbase_skill.name}")

    logger.info(f"\n✅ Total skills registered: {skills_count}")

    # Initialize Orchestrators
    logger.info("\n" + "=" * 80)
    logger.info("🎯 Initializing Orchestrators")
    logger.info("=" * 80)

    orchestrators = {}

    if llm_skill and skills_count > 1:
        # Tool Orchestrator
        logger.info("🔧 Initializing ToolOrchestrator...")
        tool_orchestrator = ToolOrchestrator(
            llm_skill=llm_skill,
            skills=agent.skills,
            db_manager=db_manager,
            use_react=False,
        )
        orchestrators["tool"] = tool_orchestrator
        logger.info("   ✅ ToolOrchestrator initialized")

        # ReAct Orchestrator
        logger.info("🔄 Initializing ToolOrchestrator (ReAct mode)...")
        react_orchestrator = ToolOrchestrator(
            llm_skill=llm_skill,
            skills=agent.skills,
            db_manager=db_manager,
            use_react=True,
        )
        orchestrators["react"] = react_orchestrator
        logger.info("   ✅ ReAct Orchestrator initialized")

        # Plan-and-Execute Orchestrator
        logger.info("📋 Initializing PlanExecuteOrchestrator...")
        plan_orchestrator = PlanExecuteOrchestrator(
            llm_skill=llm_skill, skills=agent.skills, db_manager=db_manager
        )
        orchestrators["plan_execute"] = plan_orchestrator
        logger.info("   ✅ PlanExecuteOrchestrator initialized")

        # ToT Orchestrator
        logger.info("🌳 Initializing ToTOrchestrator...")
        tot_orchestrator = ToTOrchestrator(
            llm_skill=llm_skill, skills=agent.skills, db_manager=db_manager
        )
        orchestrators["tot"] = tot_orchestrator
        logger.info("   ✅ ToTOrchestrator initialized")

        logger.info(f"\n✅ Total orchestrators initialized: {len(orchestrators)}")
        for name, orch in orchestrators.items():
            logger.info(f"   - {name}")

    # Health Check
    logger.info("\n" + "=" * 80)
    logger.info("🏥 Health Check")
    logger.info("=" * 80)
    health = agent.health_check()
    for skill_name, is_healthy in health.items():
        status = "✅" if is_healthy else "❌"
        logger.info(f"   {status} {skill_name}")

    # Self-heal if needed
    if not all(health.values()):
        logger.warning("\n⚠️  Some skills are unhealthy, attempting self-heal...")
        agent.self_heal()
        logger.info("✅ Self-heal completed")

    # Get topics from config
    topics = config.get("topics", [])
    logger.info(f"\n📚 Topics configured: {len(topics)}")
    for i, topic in enumerate(topics, 1):
        logger.info(f"   {i}. {topic}")

    # Run arXiv search workflow
    if "arxiv_search" in agent.skills and topics:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 Running arXiv Search Workflow")
        logger.info("=" * 80)

        arxiv_skill = agent.get_skill("arxiv_search")
        llm_skill = agent.get_skill("llm")
        pocketbase = agent.get_skill("pocketbase")

        for topic in topics[:3]:  # Process first 3 topics
            logger.info(f"\n🔍 Topic: {topic}")
            logger.info("-" * 80)

            # Search for papers
            logger.info("   Searching arXiv...")
            result = agent.execute_skill(
                "arxiv_search",
                query=topic,
                max_results=config.get("skills", "arxiv", "max_results"),
                months_back=config.get("skills", "arxiv", "months_back"),
            )

            if result.success:
                papers = result.data.get("papers", [])
                logger.info(f"   ✅ Found {len(papers)} papers")

                # Process each paper
                for i, paper in enumerate(papers[:3], 1):  # Process first 3
                    logger.info(f"\n   [{i}] {paper.get('title', 'Unknown')[:80]}...")
                    logger.info(f"       ArXiv ID: {paper.get('arxiv_id', 'Unknown')}")
                    logger.info(
                        f"       Published: {paper.get('published', 'Unknown')}"
                    )

                    # Summarize with LLM
                    if llm_skill:
                        logger.info("       🧠 Generating summary...")
                        summary_result = agent.execute_skill(
                            "llm",
                            prompt=f"Summarize this paper in 2-3 sentences:\n\nTitle: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}",
                            system_prompt="You are an expert research assistant. Provide concise, informative summaries.",
                            max_tokens=200,
                        )

                        if summary_result.success:
                            summary = summary_result.data.get("content", "")
                            logger.info(f"       ✅ Summary: {summary[:100]}...")
                        else:
                            logger.warning(
                                f"       ❌ Failed to generate summary: {summary_result.error}"
                            )

                        # Save to database
                        logger.info("       💾 Saving to database...")
                        try:
                            session_id = db_manager.create_session(
                                session_id=f"paper_{paper.get('arxiv_id', 'unknown')}",
                                session_type="paper_processing",
                                model=llm_skill.model,
                                orchestrator_type="Agent",
                                metadata={
                                    "topic": topic,
                                    "paper_title": paper.get("title", ""),
                                    "paper_authors": paper.get("authors", []),
                                },
                            )

                            db_manager.add_message(
                                session_id=session_id,
                                role="user",
                                content=f"Process paper: {paper.get('title', '')}",
                                message_index=0,
                                model=llm_skill.model,
                            )

                            db_manager.update_session(
                                session_id,
                                final_status="completed",
                                total_iterations=1,
                                total_tool_calls=1,
                                total_duration_ms=100,
                            )

                            logger.info(
                                f"       ✅ Saved to database (session_id: {session_id})"
                            )
                        except Exception as e:
                            logger.error(f"       ❌ Failed to save to database: {e}")

                    # Save to PocketBase
                    if pocketbase:
                        logger.info("       💾 Saving to PocketBase...")
                        save_result = agent.execute_skill(
                            "pocketbase",
                            action="save_paper",
                            arxiv_id=paper.get("arxiv_id", ""),
                            title=paper.get("title", ""),
                            authors=paper.get("authors", []),
                            published=paper.get("published", ""),
                            abstract=paper.get("abstract", ""),
                            summary=summary if summary_result.success else "",
                            url=paper.get("url", ""),
                            pdf_url=paper.get("pdf_url", ""),
                            topics=[topic],
                        )

                        if save_result.success:
                            logger.info(f"       ✅ Saved to PocketBase")
                        else:
                            logger.warning(
                                f"       ❌ Failed to save to PocketBase: {save_result.error}"
                            )
            else:
                logger.error(f"   ❌ Failed to search arXiv: {result.error}")

    # Display agent status
    logger.info("\n" + "=" * 80)
    logger.info("📊 Agent Status")
    logger.info("=" * 80)
    status = agent.get_status()
    logger.info(f"   Total Skills: {status['skills_count']}")

    total_executions = 0
    for skill in status["skills"]:
        total_executions += skill.get("usage_count", 0)
        logger.info(
            f"   - {skill.get('name', 'Unknown')}: {skill.get('usage_count', 0)} executions"
        )

    logger.info(f"   Total Executions: {total_executions}")

    # Display database statistics
    logger.info("\n" + "=" * 80)
    logger.info("🗄️  Database Statistics")
    logger.info("=" * 80)

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Count sessions
            cursor.execute("SELECT COUNT(*) FROM sessions")
            session_count = cursor.fetchone()[0]
            logger.info(f"   Sessions: {session_count}")

            # Count messages
            cursor.execute("SELECT COUNT(*) FROM messages")
            message_count = cursor.fetchone()[0]
            logger.info(f"   Messages: {message_count}")

            # Count tool calls
            cursor.execute("SELECT COUNT(*) FROM tool_calls")
            tool_call_count = cursor.fetchone()[0]
            logger.info(f"   Tool Calls: {tool_call_count}")

            # Count reasoning steps
            cursor.execute("SELECT COUNT(*) FROM reasoning_steps")
            reasoning_count = cursor.fetchone()[0]
            logger.info(f"   Reasoning Steps: {reasoning_count}")

            # Count performance metrics
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            metric_count = cursor.fetchone()[0]
            logger.info(f"   Performance Metrics: {metric_count}")

    except Exception as e:
        logger.error(f"   ❌ Failed to get database statistics: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ VibeAgent Completed Successfully!")
    logger.info("=" * 80)
    logger.info(f"📝 Full log saved to: {Path('vibeagent.log').absolute()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
