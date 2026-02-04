"""Multi-Agent Collaboration system.

This system enables specialized agents to collaborate on complex tasks,
sharing knowledge and coordinating their efforts.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .agent import Agent
from .skill import SkillResult

logger = logging.getLogger(__name__)


@dataclass
class AgentTask:
    """Task assigned to an agent."""

    task_id: str
    agent_id: str
    description: str
    parameters: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Any = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


@dataclass
class AgentMessage:
    """Message between agents."""

    message_id: str
    from_agent_id: str
    to_agent_id: str | None
    content: str
    message_type: str  # request, response, broadcast, notification
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


@dataclass
class CollaborationSession:
    """Session for agent collaboration."""

    session_id: str
    agents: list[str] = field(default_factory=list)
    tasks: list[AgentTask] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    shared_context: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None


class MultiAgentCollaboration:
    """System for multi-agent collaboration."""

    def __init__(self):
        """Initialize Multi-Agent Collaboration system."""
        self.agents: dict[str, Agent] = {}
        self.sessions: dict[str, CollaborationSession] = []
        self.message_handlers: dict[str, Any] = {}
        logger.info("MultiAgentCollaboration initialized")

    def register_agent(self, agent: Agent, role: str | None = None) -> str:
        """Register an agent for collaboration.

        agent_id: str - Unique agent ID
        """
        agent_id = agent.name
        self.agents[agent_id] = agent

        # Set agent role
        if role:
            agent.role = role

        logger.info(f"Registered agent: {agent_id} (role: {role})")
        return agent_id

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent.

        Args:
            agent_id: Agent ID to unregister

        Returns:
            Success status
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
            return True
        return False

    def create_session(
        self, agent_ids: list[str], task_description: str, shared_context: dict | None = None
    ) -> SkillResult:
        """Create a collaboration session.

        Args:
            agent_ids: List of agent IDs to participate
            task_description: Description of the collaborative task
            shared_context: Initial shared context

        Returns:
            SkillResult with session information
        """
        try:
            session_id = f"session_{datetime.now().timestamp()}"

            # Validate agents
            for agent_id in agent_ids:
                if agent_id not in self.agents:
                    return SkillResult(
                        success=False, error=f"Agent not found: {agent_id}"
                    )

            session = CollaborationSession(
                session_id=session_id,
                agents=agent_ids,
                shared_context=shared_context or {},
            )

            self.sessions[session_id] = session

            # Send notification to all agents
            self._broadcast_message(
                session_id=session_id,
                from_agent_id="system",
                content=f"New collaboration session created. Task: {task_description}",
                message_type="notification",
            )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "session_id": session_id,
                    "agents": agent_ids,
                    "task_description": task_description,
                    "shared_context": session.shared_context,
                },
            )

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return SkillResult(success=False, error=f"Session creation failed: {str(e)}")

    def assign_task(
        self, session_id: str, task_description: str, agent_id: str, parameters: dict | None = None
    ) -> SkillResult:
        """Assign a task to a specific agent.

        Args:
            session_id: Session ID
            task_description: Task description
            agent_id: Agent ID to assign task to
            parameters: Task parameters

        Returns:
            SkillResult with task assignment information
        """
        try:
            if session_id not in self.sessions:
                return SkillResult(success=False, error=f"Session not found: {session_id}")

            if agent_id not in self.agents:
                return SkillResult(success=False, error=f"Agent not found: {agent_id}")

            session = self.sessions[session_id]

            task_id = f"task_{datetime.now().timestamp()}_{len(session.tasks)}"
            task = AgentTask(
                task_id=task_id,
                agent_id=agent_id,
                description=task_description,
                parameters=parameters or {},
            )

            session.tasks.append(task)

            # Send task to agent
            self._send_message(
                session_id=session_id,
                from_agent_id="system",
                to_agent_id=agent_id,
                content=f"New task: {task_description}",
                message_type="request",
                metadata={"task_id": task_id, "parameters": parameters or {}},
            )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "description": task_description,
                    "status": task.status,
                },
            )

        except Exception as e:
            logger.error(f"Error assigning task: {e}")
            return SkillResult(success=False, error=f"Task assignment failed: {str(e)}")

    def send_message(
        self, session_id: str, from_agent_id: str, to_agent_id: str | None, content: str, message_type: str = "message"
    ) -> SkillResult:
        """Send a message between agents.

        Args:
            session_id: Session ID
            from_agent_id: Sender agent ID
            to_agent_id: Recipient agent ID (None for broadcast)
            content: Message content
            message_type: Message type

        Returns:
            SkillResult with message delivery status
        """
        try:
            if session_id not in self.sessions:
                return SkillResult(success=False, error=f"Session not found: {session_id}")

            session = self.sessions[session_id]

            if from_agent_id not in self.agents:
                return SkillResult(success=False, error=f"Sender agent not found: {from_agent_id}")

            if to_agent_id and to_agent_id not in self.agents:
                return SkillResult(success=False, error=f"Recipient agent not found: {to_agent_id}")

            message = AgentMessage(
                message_id=f"msg_{datetime.now().timestamp()}_{len(session.messages)}",
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                content=content,
                message_type=message_type,
            )

            session.messages.append(message)

            # Handle message if handler exists
            if to_agent_id and to_agent_id in self.message_handlers:
                self.message_handlers[to_agent_id](message)

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "message_id": message.message_id,
                    "from_agent_id": from_agent_id,
                    "to_agent_id": to_agent_id,
                    "status": "delivered",
                },
            )

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return SkillResult(success=False, error=f"Message delivery failed: {str(e)}")

    def _send_message(
        self,
        session_id: str,
        from_agent_id: str,
        to_agent_id: str | None,
        content: str,
        message_type: str,
        metadata: dict | None = None,
    ):
        """Internal message sending helper.

        Args:
            session_id: Session ID
            from_agent_id: Sender agent ID
            to_agent_id: Recipient agent ID
            content: Message content
            message_type: Message type
            metadata: Additional metadata
        """
        self.send_message(
            session_id=session_id,
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            content=content,
            message_type=message_type,
        )

    def _broadcast_message(
        self, session_id: str, from_agent_id: str, content: str, message_type: str = "broadcast"
    ):
        """Broadcast message to all agents in session.

        Args:
            session_id: Session ID
            from_agent_id: Sender agent ID
            content: Message content
            message_type: Message type
        """
        session = self.sessions.get(session_id)
        if session:
            for agent_id in session.agents:
                if agent_id != from_agent_id:
                    self._send_message(
                        session_id=session_id,
                        from_agent_id=from_agent_id,
                        to_agent_id=agent_id,
                        content=content,
                        message_type=message_type,
                    )

    def update_task_status(
        self, session_id: str, task_id: str, status: str, result: Any = None
    ) -> SkillResult:
        """Update task status.

        Args:
            session_id: Session ID
            task_id: Task ID
            status: New status
            result: Task result

        Returns:
            SkillResult with update status
        """
        try:
            if session_id not in self.sessions:
                return SkillResult(success=False, error=f"Session not found: {session_id}")

            session = self.sessions[session_id]

            task = next((t for t in session.tasks if t.task_id == task_id), None)
            if not task:
                return SkillResult(success=False, error=f"Task not found: {task_id}")

            task.status = status
            task.result = result

            if status == "in_progress" and not task.started_at:
                task.started_at = datetime.now().isoformat()
            elif status in ("completed", "failed") and not task.completed_at:
                task.completed_at = datetime.now().isoformat()

            # Notify agents
            self._broadcast_message(
                session_id=session_id,
                from_agent_id="system",
                content=f"Task {task_id} status updated to {status}",
                message_type="notification",
            )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "task_id": task_id,
                    "status": task.status,
                    "result": result,
                },
            )

        except Exception as e:
            logger.error(f"Error updating task status: {e}")
            return SkillResult(success=False, error=f"Status update failed: {str(e)}")

    def get_session_status(self, session_id: str) -> SkillResult:
        """Get session status.

        Args:
            session_id: Session ID

        Returns:
            SkillResult with session status
        """
        try:
            if session_id not in self.sessions:
                return SkillResult(success=False, error=f"Session not found: {session_id}")

            session = self.sessions[session_id]

            completed_tasks = len([t for t in session.tasks if t.status == "completed"])
            failed_tasks = len([t for t in session.tasks if t.status == "failed"])
            in_progress_tasks = len([t for t in session.tasks if t.status == "in_progress"])
            pending_tasks = len([t for t in session.tasks if t.status == "pending"])

            return SkillResult(
                success=True,
                data={
                    "session_id": session_id,
                    "agents": session.agents,
                    "total_tasks": len(session.tasks),
                    "completed_tasks": completed_tasks,
                    "failed_tasks": failed_tasks,
                    "in_progress_tasks": in_progress_tasks,
                    "pending_tasks": pending_tasks,
                    "total_messages": len(session.messages),
                    "created_at": session.created_at,
                    "shared_context": session.shared_context,
                },
            )

        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            return SkillResult(success=False, error=f"Status retrieval failed: {str(e)}")

    def get_agent_messages(self, session_id: str, agent_id: str) -> list[AgentMessage]:
        """Get messages for a specific agent.

        Args:
            session_id: Session ID
            agent_id: Agent ID

        Returns:
            List of messages for the agent
        """
        session = self.sessions.get(session_id)
        if session:
            return [m for m in session.messages if m.to_agent_id == agent_id or m.to_agent_id is None]
        return []

    def set_message_handler(self, agent_id: str, handler: Any):
        """Set a message handler for an agent.

        Args:
            agent_id: Agent ID
            handler: Handler function
        """
        self.message_handlers[agent_id] = handler

    def _record_usage(self):
        """Record usage statistics."""
        pass

    def get_stats(self) -> dict:
        """Get collaboration system statistics.

        Returns:
            Statistics
        """
        return {
            "total_agents": len(self.agents),
            "total_sessions": len(self.sessions),
            "active_sessions": len([s for s in self.sessions.values() if s.completed_at is None]),
            "message_handlers": len(self.message_handlers),
            "registered_agents": list(self.agents.keys()),
        }

    def health_check(self) -> bool:
        """Check if collaboration system is operational."""
        return len(self.agents) > 0