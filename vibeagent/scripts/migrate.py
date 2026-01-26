"""Database migration script for VibeAgent."""

import typer
from pathlib import Path
import sqlite3
import json
import os
from typing import Optional

app = typer.Typer(help="Database migration tools for VibeAgent")


@app.command()
def init(
    db_path: Path = typer.Option("data/vibeagent.db", "--db", help="Database file path"),
    force: bool = typer.Option(False, "--force", help="Force reinitialization"),
) -> None:
    """Initialize the database schema."""
    typer.echo(f"Initializing database at {db_path}")
    
    if db_path.exists() and not force:
        typer.confirm("Database already exists. Do you want to overwrite it?", abort=True)
    
    # Create data directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create papers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            authors TEXT NOT NULL,
            published TEXT NOT NULL,
            abstract TEXT NOT NULL,
            summary TEXT,
            url TEXT,
            pdf_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create research_sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            status TEXT NOT NULL,
            results TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create analytics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            event_type TEXT NOT NULL,
            data TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES research_sessions (id)
        )
    """)
    
    conn.commit()
    conn.close()
    
    typer.echo(f"✓ Database initialized successfully at {db_path}")


@app.command()
def migrate(
    db_path: Path = typer.Option("data/vibeagent.db", "--db", help="Database file path"),
    migrations_dir: Path = typer.Option("migrations", "--migrations-dir", help="Migrations directory"),
) -> None:
    """Run database migrations."""
    typer.echo(f"Running migrations from {migrations_dir}")
    
    if not migrations_dir.exists():
        typer.echo(f"Migrations directory {migrations_dir} does not exist", err=True)
        raise typer.Exit(1)
    
    # This is a simple migration runner - in production, use alembic
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check migrations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Get applied migrations
    cursor.execute("SELECT name FROM migrations")
    applied = {row[0] for row in cursor.fetchall()}
    
    # Apply pending migrations
    migrations = sorted(migrations_dir.glob("*.sql"))
    for migration in migrations:
        if migration.name not in applied:
            typer.echo(f"Applying {migration.name}...")
            
            with open(migration) as f:
                sql = f.read()
                cursor.executescript(sql)
            
            cursor.execute(
                "INSERT INTO migrations (name) VALUES (?)",
                (migration.name,)
            )
            typer.echo(f"✓ Applied {migration.name}")
    
    conn.commit()
    conn.close()


@app.command()
def seed(
    db_path: Path = typer.Option("data/vibeagent.db", "--db", help="Database file path"),
    seed_file: Path = typer.Option("data/seed.json", "--seed-file", help="Seed data file"),
) -> None:
    """Seed the database with initial data."""
    typer.echo(f"Seeding database from {seed_file}")
    
    if not seed_file.exists():
        typer.echo(f"Seed file {seed_file} does not exist", err=True)
        raise typer.Exit(1)
    
    with open(seed_file) as f:
        data = json.load(f)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Insert papers
    if "papers" in data:
        for paper in data["papers"]:
            cursor.execute("""
                INSERT OR IGNORE INTO papers (arxiv_id, title, authors, published, abstract, url, pdf_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                paper["arxiv_id"],
                paper["title"],
                ", ".join(paper["authors"]),
                paper["published"],
                paper["abstract"],
                paper.get("url", ""),
                paper.get("pdf_url", ""),
            ))
    
    conn.commit()
    conn.close()
    
    typer.echo("✓ Database seeded successfully")


if __name__ == "__main__":
    app()
