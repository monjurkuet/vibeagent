#!/usr/bin/env python3
"""
Database migration script for VibeAgent.

This script manages versioned database migrations with support for:
- Applying migrations incrementally
- Rollback capability
- Migration history tracking
- Migration validation
- Detailed logging
"""

import argparse
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database migrations."""

    MIGRATION_PATTERN = re.compile(r"^(\d{14})_(.+)\.sql$")

    def __init__(
        self, db_path: Optional[str] = None, migrations_dir: Optional[str] = None
    ):
        """Initialize the migration manager."""
        if db_path is None:
            db_path = "/home/muham/development/vibeagent/data/vibeagent.db"

        if migrations_dir is None:
            migrations_dir = str(Path(__file__).parent.parent / "config" / "migrations")

        self.db_path = db_path
        self.migrations_dir = Path(migrations_dir)
        self._ensure_migrations_table()

    def _ensure_migrations_table(self):
        """Create the migrations tracking table if it doesn't exist."""
        if not os.path.exists(self.db_path):
            logger.info("Database does not exist yet")
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TIMESTAMP NOT NULL,
                    rollback_sql TEXT,
                    checksum TEXT
                )
            """)
            conn.commit()

    def list_migrations(self) -> List[Dict]:
        """List all available migrations with their status."""
        migrations = self._discover_migrations()
        applied = self._get_applied_migrations()

        result = []
        for version, name, filepath in migrations:
            status = "applied" if version in applied else "pending"
            applied_at = applied.get(version, {}).get("applied_at")

            result.append(
                {
                    "version": version,
                    "name": name,
                    "status": status,
                    "applied_at": applied_at,
                    "filepath": filepath,
                }
            )

        return sorted(result, key=lambda x: x["version"])

    def get_status(self) -> Dict:
        """Get the current migration status."""
        migrations = self.list_migrations()
        applied_count = sum(1 for m in migrations if m["status"] == "applied")
        pending_count = sum(1 for m in migrations if m["status"] == "pending")

        latest_applied = None
        for m in reversed(migrations):
            if m["status"] == "applied":
                latest_applied = m["version"]
                break

        return {
            "total_migrations": len(migrations),
            "applied_count": applied_count,
            "pending_count": pending_count,
            "latest_applied": latest_applied,
            "migrations": migrations,
        }

    def migrate(self, target_version: Optional[str] = None) -> bool:
        """Apply pending migrations up to target version."""
        try:
            migrations = self._discover_migrations()
            applied = self._get_applied_migrations()

            to_apply = []
            for version, name, filepath in migrations:
                if version not in applied:
                    to_apply.append((version, name, filepath))
                    if target_version and version == target_version:
                        break
                elif target_version and version == target_version:
                    logger.info(f"Target version {target_version} already applied")
                    return True

            if not to_apply:
                logger.info("No pending migrations to apply")
                return True

            logger.info(f"Found {len(to_apply)} pending migration(s) to apply")

            for version, name, filepath in to_apply:
                if not self._apply_migration(version, name, filepath):
                    logger.error(f"Failed to apply migration {version}_{name}")
                    return False

            logger.info(f"✓ Successfully applied {len(to_apply)} migration(s)")
            return True

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

    def rollback(self, steps: int = 1, version: Optional[str] = None) -> bool:
        """Rollback migrations by steps or to a specific version."""
        try:
            applied = self._get_applied_migrations()

            if not applied:
                logger.info("No migrations to rollback")
                return True

            sorted_applied = sorted(applied.items(), key=lambda x: x[0], reverse=True)

            to_rollback = []
            if version:
                for ver, info in sorted_applied:
                    if ver <= version:
                        break
                    to_rollback.append((ver, info))
            else:
                to_rollback = sorted_applied[:steps]

            if not to_rollback:
                logger.info("No migrations to rollback")
                return True

            logger.info(f"Rolling back {len(to_rollback)} migration(s)")

            for ver, info in reversed(to_rollback):
                if not self._rollback_migration(ver, info):
                    logger.error(f"Failed to rollback migration {ver}")
                    return False

            logger.info(f"✓ Successfully rolled back {len(to_rollback)} migration(s)")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def validate_migrations(self) -> Tuple[List[str], List[str]]:
        """Validate all migration files."""
        migrations = self._discover_migrations()
        applied = self._get_applied_migrations()

        valid = []
        invalid = []

        for version, name, filepath in migrations:
            try:
                issues = self._validate_migration_file(filepath, version, name)
                if issues:
                    invalid.append(f"{version}_{name}: {', '.join(issues)}")
                else:
                    valid.append(f"{version}_{name}")
            except Exception as e:
                invalid.append(f"{version}_{name}: {str(e)}")

        return valid, invalid

    def create_migration(self, name: str) -> Optional[str]:
        """Create a new migration file with the current timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{name.replace(' ', '_').lower()}.sql"
        filepath = self.migrations_dir / filename

        template = f"""-- Migration: {name}
-- Created: {datetime.now().isoformat()}
-- Description: Add your migration description here

-- TODO: Add your UP migration SQL here
-- Example:
-- ALTER TABLE example_table ADD COLUMN new_column TEXT;

-- ROLLBACK SQL (optional)
-- Add SQL to rollback this migration below the ROLLBACK comment
-- ROLLBACK: ALTER TABLE example_table DROP COLUMN new_column;
"""

        self.migrations_dir.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write(template)

        logger.info(f"Created migration file: {filepath}")
        return str(filepath)

    def _discover_migrations(self) -> List[Tuple[str, str, Path]]:
        """Discover all migration files in the migrations directory."""
        migrations = []

        if not self.migrations_dir.exists():
            logger.warning(
                f"Migrations directory does not exist: {self.migrations_dir}"
            )
            return migrations

        for filepath in sorted(self.migrations_dir.glob("*.sql")):
            match = self.MIGRATION_PATTERN.match(filepath.name)
            if match:
                version = match.group(1)
                name = match.group(2)
                migrations.append((version, name, filepath))

        return sorted(migrations, key=lambda x: x[0])

    def _get_applied_migrations(self) -> Dict[str, Dict]:
        """Get all applied migrations from the database."""
        applied = {}

        if not os.path.exists(self.db_path):
            return applied

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT version, name, applied_at, rollback_sql, checksum FROM schema_migrations"
            )
            for row in cursor:
                applied[row[0]] = {
                    "name": row[1],
                    "applied_at": row[2],
                    "rollback_sql": row[3],
                    "checksum": row[4],
                }

        return applied

    def _apply_migration(self, version: str, name: str, filepath: Path) -> bool:
        """Apply a single migration."""
        logger.info(f"Applying migration: {version}_{name}")

        with open(filepath, "r") as f:
            migration_sql = f.read()

        up_sql, rollback_sql = self._extract_migration_sql(migration_sql)
        checksum = self._calculate_checksum(migration_sql)

        if not up_sql.strip():
            logger.error(f"No SQL found in migration: {version}_{name}")
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN TRANSACTION")

                try:
                    conn.executescript(up_sql)

                    conn.execute(
                        """
                        INSERT INTO schema_migrations (version, name, applied_at, rollback_sql, checksum)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            version,
                            name,
                            datetime.now().isoformat(),
                            rollback_sql,
                            checksum,
                        ),
                    )

                    conn.commit()
                    logger.info(f"✓ Applied migration: {version}_{name}")
                    return True

                except Exception as e:
                    conn.rollback()
                    raise

        except Exception as e:
            logger.error(f"Failed to apply migration {version}_{name}: {e}")
            return False

    def _rollback_migration(self, version: str, info: Dict) -> bool:
        """Rollback a single migration."""
        logger.info(f"Rolling back migration: {version}_{info['name']}")

        rollback_sql = info.get("rollback_sql")

        if not rollback_sql or not rollback_sql.strip():
            logger.error(f"No rollback SQL available for migration: {version}")
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN TRANSACTION")

                try:
                    conn.executescript(rollback_sql)

                    conn.execute(
                        "DELETE FROM schema_migrations WHERE version = ?", (version,)
                    )

                    conn.commit()
                    logger.info(f"✓ Rolled back migration: {version}")
                    return True

                except Exception as e:
                    conn.rollback()
                    raise

        except Exception as e:
            logger.error(f"Failed to rollback migration {version}: {e}")
            return False

    def _extract_migration_sql(self, content: str) -> Tuple[str, str]:
        """Extract UP and ROLLBACK SQL from migration file."""
        rollback_marker = "\n-- ROLLBACK:"

        if rollback_marker in content:
            parts = content.split(rollback_marker, 1)
            up_sql = parts[0]
            rollback_sql = parts[1].strip()
        else:
            up_sql = content
            rollback_sql = ""

        return up_sql, rollback_sql

    def _calculate_checksum(self, content: str) -> str:
        """Calculate a simple checksum for the migration content."""
        import hashlib

        return hashlib.md5(content.encode()).hexdigest()

    def _validate_migration_file(
        self, filepath: Path, version: str, name: str
    ) -> List[str]:
        """Validate a migration file."""
        issues = []

        with open(filepath, "r") as f:
            content = f.read()

        up_sql, rollback_sql = self._extract_migration_sql(content)

        if not up_sql.strip():
            issues.append("No UP migration SQL found")

        if "-- ROLLBACK:" in content and not rollback_sql.strip():
            issues.append("ROLLBACK marker present but no rollback SQL")

        return issues


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Manage database migrations for VibeAgent"
    )

    subparsers = parser.add_subparsers(dest="command", help="Migration commands")

    parser.add_argument("--db-path", type=str, help="Path to the database file")
    parser.add_argument(
        "--migrations-dir", type=str, help="Path to the migrations directory"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    migrate_cmd = subparsers.add_parser("migrate", help="Apply pending migrations")
    migrate_cmd.add_argument(
        "--to", type=str, dest="to_version", help="Target version to migrate to"
    )

    rollback_cmd = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_cmd.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of migrations to rollback (default: 1)",
    )
    rollback_cmd.add_argument(
        "--to", type=str, dest="to_version", help="Rollback to this version"
    )

    subparsers.add_parser("status", help="Show migration status")
    subparsers.add_parser("list", help="List all migrations")

    validate_cmd = subparsers.add_parser("validate", help="Validate migration files")

    create_cmd = subparsers.add_parser("create", help="Create a new migration")
    create_cmd.add_argument("name", type=str, help="Name/description of the migration")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    manager = MigrationManager(db_path=args.db_path, migrations_dir=args.migrations_dir)

    if args.command == "status":
        status = manager.get_status()
        print(f"\nMigration Status:")
        print(f"  Total: {status['total_migrations']}")
        print(f"  Applied: {status['applied_count']}")
        print(f"  Pending: {status['pending_count']}")
        print(f"  Latest: {status['latest_applied'] or 'None'}")

        if status["migrations"]:
            print(f"\nMigrations:")
            for m in status["migrations"]:
                status_symbol = "✓" if m["status"] == "applied" else "○"
                print(f"  {status_symbol} {m['version']} - {m['name']}")
                if m["applied_at"]:
                    print(f"    Applied: {m['applied_at']}")

    elif args.command == "list":
        migrations = manager.list_migrations()
        print(f"\nMigrations ({len(migrations)} total):\n")
        for m in migrations:
            status_symbol = "✓" if m["status"] == "applied" else "○"
            print(f"  {status_symbol} {m['version']} - {m['name']}")
            print(f"    Status: {m['status']}")
            if m["applied_at"]:
                print(f"    Applied: {m['applied_at']}")
            print()

    elif args.command == "migrate":
        success = manager.migrate(target_version=args.to_version)
        sys.exit(0 if success else 1)

    elif args.command == "rollback":
        if args.to_version:
            success = manager.rollback(version=args.to_version)
        else:
            success = manager.rollback(steps=args.steps)
        sys.exit(0 if success else 1)

    elif args.command == "validate":
        valid, invalid = manager.validate_migrations()

        print(f"\nValidation Results:")
        print(f"  Valid: {len(valid)}")
        print(f"  Invalid: {len(invalid)}")

        if invalid:
            print(f"\nInvalid migrations:")
            for m in invalid:
                print(f"  ✗ {m}")

        if valid:
            print(f"\nValid migrations:")
            for m in valid:
                print(f"  ✓ {m}")

        sys.exit(0 if not invalid else 1)

    elif args.command == "create":
        filepath = manager.create_migration(args.name)
        if filepath:
            print(f"\n✓ Created migration: {filepath}")
            sys.exit(0)
        else:
            print(f"\n✗ Failed to create migration")
            sys.exit(1)


if __name__ == "__main__":
    main()
