"""
FastAPI backend for VibeAgent Dashboard.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Agent
from core.hybrid_orchestrator import HybridOrchestrator
from skills import ArxivSkill, ScraperSkill, LLMSkill, SqliteSkill
from config import Config

# Global agent instance
agent = None
hybrid_orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global agent, hybrid_orchestrator

    # Load configuration (use absolute path from project root)
    project_root = Path(__file__).parent.parent
    config = Config(config_path=str(project_root / "config" / "agent_config.json"))
    agent = Agent(name=config.get("agent", "name"))

    # Register skills
    if config.get("skills", "arxiv", "enabled"):
        agent.register_skill(ArxivSkill())

    if config.get("skills", "scraper", "enabled"):
        agent.register_skill(ScraperSkill())

    llm_skill = None
    if config.get("skills", "llm", "enabled"):
        llm_skill = LLMSkill(
            base_url=config.get("skills", "llm", "base_url"),
            model=config.get("skills", "llm", "model"),
        )
        agent.register_skill(llm_skill)

    if config.get("sqlite", "database_path"):
        agent.register_skill(
            SqliteSkill(
                db_path=config.get("sqlite", "database_path"),
            )
        )

    print(f"🤖 {agent.name} initialized with {len(agent.skills)} skills")

    # Initialize HybridOrchestrator if LLM skill is available
    if llm_skill and len(agent.skills) > 1:
        hybrid_orchestrator = HybridOrchestrator(llm_skill, agent.skills)
        print(f"🔧 HybridOrchestrator initialized with tool calling support")

    yield

    # Cleanup on shutdown
    print("👋 Shutting down...")


app = FastAPI(
    title="VibeAgent API",
    description="API for VibeAgent Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "VibeAgent API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if agent:
        health = agent.health_check()
        # Convert health dict to array of skill info objects for frontend
        skills_info = []
        for skill_name, skill in agent.skills.items():
            skill_info = skill.get_info()
            # Add 'healthy' property for frontend compatibility
            skill_info["healthy"] = health.get(skill_name, False)
            skills_info.append(skill_info)
        return {"status": "healthy", "agent": agent.name, "skills": skills_info}
    return {"status": "initializing"}


@app.get("/agent/skills")
async def get_skills():
    """Get list of available skills."""
    if not agent:
        return {"skills": []}

    status = agent.get_status()
    return {"skills": status["skills"]}


@app.get("/agent/status")
async def get_agent_status():
    """Get detailed agent status."""
    if not agent:
        return {"status": "not_initialized"}

    status = agent.get_status()
    return status


@app.post("/agent/prompt")
async def process_prompt(request: dict):
    """Process a detailed prompt and break it down into tasks."""
    if not agent:
        return {"error": "Agent not initialized"}

    prompt = request.get("prompt", "")
    if not prompt:
        return {"error": "Prompt is required"}

    # Use LLM to break down the prompt into subtasks
    llm_skill = agent.get_skill("llm")
    if not llm_skill:
        return {"error": "LLM skill not available"}

    breakdown_prompt = f"""
    Analyze this user request and break it down into specific, actionable subtasks.
    For each subtask, identify:
    1. The task description
    2. Which skill(s) to use (arxiv_search, scraper, llm, pocketbase)
    3. Required parameters
    4. Dependencies on other tasks

    User request: {prompt}

    Return as JSON with this structure:
    {{
        "main_goal": "brief description",
        "subtasks": [
            {{
                "id": "task_1",
                "description": "task description",
                "skill": "skill_name",
                "parameters": {{}},
                "dependencies": []
            }}
        ]
    }}
    """

    result = agent.execute_skill(
        "llm",
        prompt=breakdown_prompt,
        system_prompt="You are a task planning assistant. Respond only with valid JSON.",
        max_tokens=2000,
    )

    if result.success:
        return {"success": True, "breakdown": result.data["content"]}
    else:
        return {"success": False, "error": result.error}


@app.post("/agent/execute")
async def execute_skill(request: dict):
    """Execute a specific skill."""
    if not agent:
        return {"error": "Agent not initialized"}

    skill_name = request.get("skill")
    parameters = request.get("parameters", {})

    result = agent.execute_skill(skill_name, **parameters)

    return {"success": result.success, "data": result.data, "error": result.error}


@app.post("/agent/chat_with_tools")
async def chat_with_tools(request: dict):
    """Chat with the agent using HybridOrchestrator for tool calling."""
    try:
        if not hybrid_orchestrator:
            return {
                "error": "HybridOrchestrator not initialized",
                "message": "LLM skill or additional skills not configured",
            }

        message = request.get("message", "")
        if not message:
            return {"error": "Message is required"}

        max_iterations = request.get("max_iterations", 10)

        result = hybrid_orchestrator.execute(message, max_iterations=max_iterations)

        return {
            "success": result.success,
            "message": result.final_response,
            "method_used": result.method_used,
            "iterations": result.iterations,
            "tool_calls_made": result.tool_calls_made,
            "tool_results": result.tool_results,
            "error": result.error,
            "metadata": result.metadata,
        }

    except Exception as e:
        return {"success": False, "error": f"Error processing request: {str(e)}"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)

    try:
        await websocket.send_json(
            {"type": "connected", "message": "WebSocket connection established"}
        )
    except Exception as e:
        print(f"Error sending initial message: {e}")

    try:
        while True:
            data = await websocket.receive_json()

            # Handle different message types
            if data.get("type") == "message":
                use_hybrid = data.get("use_hybrid", False)

                if use_hybrid and hybrid_orchestrator:
                    try:
                        result = hybrid_orchestrator.execute(
                            data.get("content", ""), max_iterations=10
                        )

                        response = {
                            "type": "message",
                            "role": "assistant",
                            "content": result.final_response,
                            "method_used": result.method_used,
                            "tool_calls_made": result.tool_calls_made,
                        }
                        await manager.broadcast(response)
                    except Exception as e:
                        print(f"Hybrid orchestrator error: {e}")
                        response = {
                            "type": "message",
                            "role": "assistant",
                            "content": f"Error: {str(e)}",
                        }
                        await manager.broadcast(response)

                elif agent and "llm" in agent.skills:
                    llm_result = agent.execute_skill(
                        "llm",
                        prompt=data.get("content", ""),
                        system_prompt="You are a helpful AI assistant.",
                        max_tokens=1000,
                    )

                    response = {
                        "type": "message",
                        "role": "assistant",
                        "content": llm_result.data["content"]
                        if llm_result.success
                        else "I'm sorry, I couldn't process that.",
                    }
                    await manager.broadcast(response)

            elif data.get("type") == "status":
                # Send agent status
                if agent:
                    status = agent.get_status()
                    await manager.broadcast({"type": "status", "data": status})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    from config import Config

    config = Config()
    port = config.get_api_port()

    print(f"🚀 Starting API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
