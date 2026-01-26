"""CLI for VibeAgent."""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
import json

from . import __version__
from .core.agent import Agent
from .skills import ArxivSkill, ScraperSkill, LLMSkill, SqliteSkill
from .config import Config

app = typer.Typer(
    name="vibeagent",
    help="VibeAgent - A clean agentic system for searching and summarizing arXiv papers",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Show version information."""
    if value:
        console.print(f"VibeAgent v{__version__}")
        raise typer.Exit()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query for arXiv papers"),
    max_results: int = typer.Option(10, "--max-results", "-n", help="Maximum number of results"),
    config_path: Path = typer.Option("config/agent_config.json", "--config", "-c", help="Path to config file"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path (JSON)"),
) -> None:
    """Search for papers on arXiv."""
    console.print(f"[blue]Searching arXiv for:[/blue] {query}")
    
    # Initialize agent
    config = Config(config_path=str(config_path))
    agent = Agent(config)
    
    # Configure skills
    arxiv_skill = ArxivSkill(
        max_results=max_results,
        months_back=config.config["skills"]["arxiv"]["months_back"]
    )
    llm_skill = LLMSkill(
        base_url=config.config["skills"]["llm"]["base_url"],
        model=config.config["skills"]["llm"]["model"]
    )
    
    agent.add_skill(arxiv_skill)
    agent.add_skill(llm_skill)
    
    # Execute search
    try:
        papers = arxiv_skill.search_papers(query)
        console.print(f"[green]Found {len(papers)} papers[/green]")
        
        # Summarize papers
        console.print("[blue]Summarizing papers...[/blue]")
        for i, paper in enumerate(papers, 1):
            console.print(f"\n[yellow]{i}. {paper.title}[/yellow]")
            console.print(f"   Authors: {', '.join(paper.authors)}")
            summary = llm_skill.summarize_paper(paper)
            console.print(f"   Summary: {summary[:200]}...")
        
        # Save output if requested
        if output:
            with open(output, 'w') as f:
                json.dump([{"title": p.title, "authors": p.authors, "summary": p.summary} for p in papers], f, indent=2)
            console.print(f"\n[green]Results saved to {output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Dashboard host"),
    port: int = typer.Option(8080, "--port", "-p", help="Dashboard port"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
) -> None:
    """Launch the analytics dashboard."""
    console.print(f"[blue]Starting dashboard on[/blue] http://{host}:{port}")
    
    try:
        import uvicorn
        uvicorn.run(
            "vibeagent.core.analytics_dashboard:app",
            host=host,
            port=port,
            reload=debug,
        )
    except ImportError:
        console.print("[red]Error: Dashboard dependencies not installed[/red]")
        raise typer.Exit(1)


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=version_callback, is_eager=True),
) -> None:
    """VibeAgent - A clean agentic system for searching and summarizing arXiv papers."""
    pass


if __name__ == "__main__":
    app()
