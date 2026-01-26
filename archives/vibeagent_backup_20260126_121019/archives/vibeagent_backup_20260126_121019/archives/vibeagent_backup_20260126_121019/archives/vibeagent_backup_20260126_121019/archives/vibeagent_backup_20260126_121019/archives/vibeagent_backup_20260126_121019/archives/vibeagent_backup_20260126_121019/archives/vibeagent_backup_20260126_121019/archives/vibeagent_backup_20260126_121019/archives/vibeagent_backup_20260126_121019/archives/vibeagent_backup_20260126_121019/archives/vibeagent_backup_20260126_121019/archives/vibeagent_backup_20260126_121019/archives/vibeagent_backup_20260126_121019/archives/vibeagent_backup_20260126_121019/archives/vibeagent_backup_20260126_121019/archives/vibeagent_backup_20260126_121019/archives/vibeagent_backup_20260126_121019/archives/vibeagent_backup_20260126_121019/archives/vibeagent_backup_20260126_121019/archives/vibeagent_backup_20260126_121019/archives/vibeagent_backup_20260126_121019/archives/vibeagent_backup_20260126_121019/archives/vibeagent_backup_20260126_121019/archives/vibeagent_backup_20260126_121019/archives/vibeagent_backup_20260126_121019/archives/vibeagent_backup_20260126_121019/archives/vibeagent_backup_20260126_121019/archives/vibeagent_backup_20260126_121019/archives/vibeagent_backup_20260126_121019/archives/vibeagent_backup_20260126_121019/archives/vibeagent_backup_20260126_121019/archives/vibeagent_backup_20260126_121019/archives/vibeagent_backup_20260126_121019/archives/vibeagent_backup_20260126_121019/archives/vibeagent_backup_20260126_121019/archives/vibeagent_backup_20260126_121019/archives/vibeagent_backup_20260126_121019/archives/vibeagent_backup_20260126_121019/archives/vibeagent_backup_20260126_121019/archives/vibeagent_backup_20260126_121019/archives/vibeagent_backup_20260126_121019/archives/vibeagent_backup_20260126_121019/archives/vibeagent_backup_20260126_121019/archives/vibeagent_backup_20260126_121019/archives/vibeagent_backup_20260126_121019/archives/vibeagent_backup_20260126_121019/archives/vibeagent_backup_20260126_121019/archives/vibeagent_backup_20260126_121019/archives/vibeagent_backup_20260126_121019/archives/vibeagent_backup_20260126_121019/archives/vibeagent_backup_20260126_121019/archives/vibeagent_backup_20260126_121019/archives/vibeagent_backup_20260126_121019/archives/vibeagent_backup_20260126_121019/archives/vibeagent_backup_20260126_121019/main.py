"""Main entry point for the VibeAgent framework."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core import Agent
from skills import ArxivSkill, ScraperSkill, LLMSkill, PocketBaseSkill
from config import Config


def main():
    """Main entry point."""
    # Load configuration
    config = Config()

    # Initialize agent
    agent = Agent(name=config.get("agent", "name"))

    print(f"🤖 {agent.name} Framework Starting...")
    print(f"   Version: {config.get('agent', 'version')}\n")

    # Register skills
    if config.get("skills", "arxiv", "enabled"):
        arxiv_skill = ArxivSkill()
        agent.register_skill(arxiv_skill)

    if config.get("skills", "scraper", "enabled"):
        scraper_skill = ScraperSkill()
        agent.register_skill(scraper_skill)

    if config.get("skills", "llm", "enabled"):
        llm_skill = LLMSkill(
            base_url=config.get("skills", "llm", "base_url"),
            model=config.get("skills", "llm", "model")
        )
        agent.register_skill(llm_skill)

    if config.get("pocketbase", "url"):
        pocketbase_skill = PocketBaseSkill(
            base_url=config.get("pocketbase", "url"),
            email=config.get("pocketbase", "email"),
            password=config.get("pocketbase", "password")
        )
        agent.register_skill(pocketbase_skill)

    # Health check
    print("🏥 Running health check...")
    health = agent.health_check()
    for skill_name, is_healthy in health.items():
        status = "✅" if is_healthy else "❌"
        print(f"   {status} {skill_name}")

    # Self-heal if needed
    if not all(health.values()):
        print()
        agent.self_heal()

    # Get topics from config
    topics = config.get("topics", [])

    # Run arXiv search workflow
    if "arxiv_search" in agent.skills and topics:
        print(f"\n📚 Searching arXiv for {len(topics)} topics...\n")

        arxiv_skill = agent.get_skill("arxiv_search")
        llm_skill = agent.get_skill("llm")
        pocketbase = agent.get_skill("pocketbase")

        for topic in topics:
            print(f"🔍 Topic: {topic}")

            # Search for papers
            result = agent.execute_skill(
                "arxiv_search",
                query=topic,
                max_results=config.get("skills", "arxiv", "max_results"),
                months_back=config.get("skills", "arxiv", "months_back")
            )

            if result.success:
                papers = result.data["papers"]
                print(f"   Found {len(papers)} papers")

                # Process each paper
                for i, paper in enumerate(papers[:5], 1):  # Process first 5
                    print(f"   [{i}] {paper['title'][:60]}...")

                    # Summarize with LLM
                    if llm_skill:
                        summary_result = agent.execute_skill(
                            "llm",
                            prompt=f"Summarize this paper: {paper['title']}\n\nAbstract: {paper['abstract']}",
                            system_prompt="You are an expert research assistant.",
                            max_tokens=500
                        )

                        summary = summary_result.data["content"] if summary_result.success else ""

                        # Save to PocketBase
                        if pocketbase:
                            save_result = agent.execute_skill(
                                "pocketbase",
                                action="save_paper",
                                arxiv_id=paper["arxiv_id"],
                                title=paper["title"],
                                authors=paper["authors"],
                                published=paper["published"],
                                abstract=paper["abstract"],
                                summary=summary,
                                url=paper["url"],
                                pdf_url=paper["pdf_url"],
                                topics=[topic]
                            )

                            if save_result.success:
                                print(f"       💾 Saved to PocketBase")
                            else:
                                print(f"       ❌ Failed to save: {save_result.error}")

            print()

    # Display agent status
    print("📊 Agent Status:")
    status = agent.get_status()
    print(f"   Skills: {status['skills_count']}")
    print(f"   Total executions: {sum(s['usage_count'] for s in status['skills'])}")

    print(f"\n✅ {agent.name} completed successfully!")


if __name__ == "__main__":
    main()