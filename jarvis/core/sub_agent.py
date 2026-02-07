"""
Lightweight sub-agent for parallel task execution.

A sub-agent is a minimal agent with a limited tool set
that handles a single subtask independently.
"""

import asyncio
import time
from typing import List, Optional
from dataclasses import dataclass

from .tool_executor import AsyncToolExecutor, ToolResult


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""
    task: str
    response: str
    tools_used: List[ToolResult]
    duration: float
    success: bool
    error: Optional[str] = None


class SubAgent:
    """
    Lightweight agent for handling a single subtask.

    Gets its own tool set and context but shares the LLM provider.
    Reports progress back to the orchestrator.
    """

    def __init__(self, provider, tools: dict, context: str = ""):
        self.provider = provider
        self.tool_executor = AsyncToolExecutor(tools)
        self.context = context
        self.tools_used: List[ToolResult] = []

    async def run(self, task: str, system: str = "") -> SubAgentResult:
        """
        Execute a single subtask.

        Args:
            task: The task to complete
            system: System prompt

        Returns:
            SubAgentResult with response and tools used
        """
        start = time.time()

        system_prompt = system or "You are a helpful assistant. Complete the given task concisely."
        if self.context:
            system_prompt += f"\n\nContext:\n{self.context}"

        try:
            from jarvis.providers import Message

            messages = [Message(role="user", content=task)]

            # Simple single-turn: just get a response
            response = await asyncio.to_thread(
                self.provider.chat,
                messages=messages,
                system=system_prompt,
                stream=False,
            )

            # Handle generator vs string response
            if hasattr(response, '__iter__') and not isinstance(response, str):
                response = ''.join(str(c) for c in response)

            duration = time.time() - start

            return SubAgentResult(
                task=task,
                response=str(response).strip(),
                tools_used=self.tools_used,
                duration=duration,
                success=True,
            )

        except Exception as e:
            duration = time.time() - start
            return SubAgentResult(
                task=task,
                response="",
                tools_used=self.tools_used,
                duration=duration,
                success=False,
                error=str(e),
            )
