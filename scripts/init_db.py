#!/usr/bin/env python3
"""
Database initialization script for VibeAgent.

This script creates the data directory, initializes the database schema,
and seeds initial data including default configurations, test cases, and prompts.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import sqlite3


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handles database initialization and seeding."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database manager."""
        if db_path is None:
            db_path = "/home/muham/development/vibeagent/data/vibeagent.db"

        self.db_path = db_path
        self.project_root = Path(__file__).parent.parent

    def initialize(self, force: bool = False, skip_seeds: bool = False) -> bool:
        """Initialize the database with schema and seed data."""
        try:
            logger.info(f"Initializing database at {self.db_path}")

            if force and os.path.exists(self.db_path):
                logger.warning(
                    f"Force mode: Removing existing database at {self.db_path}"
                )
                os.remove(self.db_path)

            self._create_data_directory()
            self._initialize_schema()

            if not skip_seeds:
                self._seed_default_configurations()
                self._seed_test_cases()
                self._seed_default_prompts()

            self._verify_database()

            logger.info("✓ Database initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"✗ Database initialization failed: {e}")
            return False

    def _create_data_directory(self):
        """Create the data directory if it doesn't exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            logger.info(f"Creating data directory: {db_dir}")
            os.makedirs(db_dir, exist_ok=True)
            logger.info("✓ Data directory created")
        else:
            logger.info("✓ Data directory exists")

    def _initialize_schema(self):
        """Initialize the database schema from schema.sql."""
        schema_path = self.project_root / "config" / "schema.sql"

        if not schema_path.exists():
            logger.error(f"Schema file not found: {schema_path}")
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        logger.info(f"Initializing schema from {schema_path}")

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema_sql)
            conn.commit()

        logger.info("✓ Schema initialized successfully")

    def _seed_default_configurations(self):
        """Seed default configuration values."""
        logger.info("Seeding default configurations")

        configs = [
            {
                "key": "agent_name",
                "value": "VibeAgent",
                "description": "Default agent name",
            },
            {"key": "agent_version", "value": "1.0.0", "description": "Agent version"},
            {
                "key": "max_iterations",
                "value": "10",
                "description": "Maximum iterations for hybrid orchestrator",
            },
            {
                "key": "default_papers_limit",
                "value": "50",
                "description": "Default limit for listing papers",
            },
            {
                "key": "log_interactions",
                "value": "true",
                "description": "Enable logging of agent interactions",
            },
        ]

        with sqlite3.connect(self.db_path) as conn:
            for config in configs:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO configurations (key, value, description, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (config["key"], config["value"], config["description"]),
                )
            conn.commit()

        logger.info(f"✓ Seeded {len(configs)} default configurations")

    def _seed_test_cases(self):
        """Seed test cases from tests/test_cases.py."""
        logger.info("Seeding test cases")

        test_cases_path = self.project_root / "tests" / "test_cases.py"

        if not test_cases_path.exists():
            logger.warning(f"Test cases file not found: {test_cases_path}")
            return

        sys.path.insert(0, str(self.project_root))
        from tests.test_cases import TEST_CASES

        with sqlite3.connect(self.db_path) as conn:
            seeded_count = 0
            for test_case in TEST_CASES:
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO test_cases 
                        (name, description, messages_json, tools_json, 
                         expected_tools_json, expected_params_json, expect_no_tools, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (
                            test_case.get("name", ""),
                            test_case.get("description", ""),
                            json.dumps(test_case.get("messages", [])),
                            json.dumps(test_case.get("tools", [])),
                            json.dumps(test_case.get("expected_tool"))
                            if test_case.get("expected_tool")
                            else None,
                            json.dumps(test_case.get("expected_params"))
                            if test_case.get("expected_params")
                            else None,
                            1 if test_case.get("expect_no_tools") else 0,
                        ),
                    )
                    seeded_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to seed test case '{test_case.get('name', 'unknown')}': {e}"
                    )

            conn.commit()

        logger.info(f"✓ Seeded {seeded_count} test cases")

    def _seed_default_prompts(self):
        """Seed default system prompts."""
        logger.info("Seeding default prompts")

        prompts = [
            {
                "name": "default_assistant",
                "description": "Default helpful assistant prompt",
                "content": "You are a helpful AI assistant. Provide clear, accurate, and concise responses.",
                "category": "assistant",
            },
            {
                "name": "task_planner",
                "description": "Prompt for breaking down tasks into subtasks",
                "content": """You are a task planning assistant. Analyze the user request and break it down into specific, actionable subtasks.
For each subtask, identify:
1. The task description
2. Which skill(s) to use
3. Required parameters
4. Dependencies on other tasks

Return your response as valid JSON.""",
                "category": "planning",
            },
            {
                "name": "arxiv_researcher",
                "description": "Prompt for arXiv research tasks",
                "content": "You are a research assistant specializing in academic papers. Help users search, analyze, and summarize research papers from arXiv.",
                "category": "research",
            },
            {
                "name": "code_analyzer",
                "description": "Prompt for code analysis tasks",
                "content": "You are a code analysis assistant. Help users understand, debug, and improve their code.",
                "category": "code",
            },
        ]

        with sqlite3.connect(self.db_path) as conn:
            for prompt in prompts:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO prompts (name, description, content, category, is_active, updated_at)
                    VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
                    """,
                    (
                        prompt["name"],
                        prompt["description"],
                        prompt["content"],
                        prompt["category"],
                    ),
                )
            conn.commit()

        logger.info(f"✓ Seeded {len(prompts)} default prompts")

    def _verify_database(self):
        """Verify that the database was created correctly."""
        logger.info("Verifying database creation")

        with sqlite3.connect(self.db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()

            expected_tables = {
                "papers",
                "prompts",
                "test_cases",
                "configurations",
                "agent_interactions",
            }

            table_names = {t[0] for t in tables}
            missing_tables = expected_tables - table_names

            if missing_tables:
                raise Exception(f"Missing tables: {missing_tables}")

            config_count = conn.execute(
                "SELECT COUNT(*) FROM configurations"
            ).fetchone()[0]
            prompt_count = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
            test_case_count = conn.execute(
                "SELECT COUNT(*) FROM test_cases"
            ).fetchone()[0]

            logger.info("✓ Database verification passed")
            logger.info(f"  - Tables: {len(table_names)}")
            logger.info(f"  - Configurations: {config_count}")
            logger.info(f"  - Prompts: {prompt_count}")
            logger.info(f"  - Test cases: {test_case_count}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Initialize the VibeAgent database with schema and seed data"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to the database file (default: /home/muham/development/vibeagent/data/vibeagent.db)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-initialization by deleting existing database",
    )
    parser.add_argument(
        "--skip-seeds",
        action="store_true",
        help="Skip seeding default data (configurations, test cases, prompts)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    initializer = DatabaseInitializer(db_path=args.db_path)
    success = initializer.initialize(force=args.force, skip_seeds=args.skip_seeds)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
