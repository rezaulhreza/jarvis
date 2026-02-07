"""
Multi-agent orchestrator for complex task decomposition.

Breaks complex requests into parallel subtasks handled by sub-agents,
then merges results into a coherent response.
"""

import asyncio
import time
import re
import json
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass, field

from .sub_agent import SubAgent, SubAgentResult


@dataclass
class SubTask:
    """A decomposed subtask for parallel execution."""
    id: int
    description: str
    depends_on: List[int] = field(default_factory=list)
    tools_needed: List[str] = field(default_factory=list)
    result: Optional[SubAgentResult] = None


@dataclass
class OrchestratorEvent:
    """Event emitted during orchestration."""
    type: str  # 'decompose', 'subtask_start', 'subtask_complete', 'merge', 'done'
    subtask_id: int = 0
    subtask_description: str = ""
    result: str = ""
    total_subtasks: int = 0


@dataclass
class OrchestratorResult:
    """Final result from orchestration."""
    response: str
    subtasks: List[SubTask]
    total_duration: float
    success: bool


class AgentOrchestrator:
    """
    Orchestrates multiple sub-agents for complex tasks.

    Flow:
    1. LLM decomposes complex request into subtasks
    2. Independent subtasks run in parallel via sub-agents
    3. Dependent subtasks wait for their dependencies
    4. Results are merged into a final response

    Usage:
        orchestrator = AgentOrchestrator(provider)
        async for event in orchestrator.run("Analyze PDF and find related articles"):
            if event.type == "subtask_start":
                print(f"Starting: {event.subtask_description}")
    """

    # Patterns that suggest decomposition would help
    COMPLEXITY_PATTERNS = [
        r'\band\b.*\band\b',           # Multiple "and" clauses
        r'\bthen\b',                     # Sequential operations
        r'\bcompare\b.*\bwith\b',       # Comparison tasks
        r'\b(?:first|second|third)\b',  # Numbered steps
        r'\bboth\b',                     # Multiple targets
        r'\ball\b.*\bfiles?\b',         # Multiple file operations
        r'\banalyze\b.*\bsearch\b',     # Multi-skill tasks
    ]

    def __init__(self, provider, tool_map: dict = None, config: dict = None):
        self.provider = provider
        self.tool_map = tool_map or {}
        self.config = config or {}

    @classmethod
    def should_orchestrate(cls, message: str) -> bool:
        """Check if a message would benefit from multi-agent decomposition."""
        msg_lower = message.lower()

        # Check complexity patterns
        pattern_matches = sum(1 for p in cls.COMPLEXITY_PATTERNS if re.search(p, msg_lower))
        if pattern_matches >= 2:
            return True

        # Check message length (long messages often have multiple tasks)
        if len(message) > 300 and pattern_matches >= 1:
            return True

        return False

    async def decompose(self, task: str) -> List[SubTask]:
        """
        Use LLM to break a complex task into subtasks.

        Returns list of SubTask objects with dependency info.
        """
        prompt = f"""Break this task into 2-4 independent subtasks that can be done in parallel.
Output ONLY valid JSON array. Each item has: "id" (int), "description" (string), "depends_on" (array of ids).

Task: {task}

JSON:"""

        try:
            from jarvis.providers import Message

            response = await asyncio.to_thread(
                self.provider.chat,
                messages=[Message(role="user", content=prompt)],
                system="You decompose tasks into subtasks. Output ONLY JSON.",
                stream=False,
            )

            if hasattr(response, '__iter__') and not isinstance(response, str):
                response = ''.join(str(c) for c in response)

            text = str(response).strip()

            # Extract JSON from response
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                subtasks = []
                for item in data:
                    subtasks.append(SubTask(
                        id=item.get("id", len(subtasks) + 1),
                        description=item.get("description", ""),
                        depends_on=item.get("depends_on", []),
                    ))
                return subtasks

        except Exception as e:
            print(f"[Orchestrator] Decomposition failed: {e}")

        # Fallback: treat as single task
        return [SubTask(id=1, description=task)]

    async def execute(self, subtasks: List[SubTask]) -> List[SubAgentResult]:
        """Execute subtasks, respecting dependencies."""
        results: dict[int, SubAgentResult] = {}
        pending = list(subtasks)

        while pending:
            # Find subtasks whose dependencies are met
            ready = [
                st for st in pending
                if all(dep in results for dep in st.depends_on)
            ]

            if not ready:
                # Deadlock - force execute remaining
                ready = pending[:]

            # Execute ready subtasks in parallel
            async def _run_subtask(st: SubTask) -> tuple:
                # Build context from dependencies
                dep_context = ""
                for dep_id in st.depends_on:
                    if dep_id in results:
                        dep_result = results[dep_id]
                        dep_context += f"\nResult from subtask {dep_id}: {dep_result.response[:500]}\n"

                agent = SubAgent(
                    provider=self.provider,
                    tools=self.tool_map,
                    context=dep_context,
                )
                result = await agent.run(st.description)
                return st.id, result

            tasks = [_run_subtask(st) for st in ready]
            completed = await asyncio.gather(*tasks)

            for task_id, result in completed:
                results[task_id] = result

            # Remove completed from pending
            ready_ids = {st.id for st in ready}
            pending = [st for st in pending if st.id not in ready_ids]

        return [results[st.id] for st in subtasks if st.id in results]

    async def merge_results(self, task: str, results: List[SubAgentResult]) -> str:
        """Merge sub-agent results into a coherent response."""
        if len(results) == 1:
            return results[0].response

        # Build summary of all results
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"Subtask {i} ({r.task[:50]}):\n{r.response[:500]}")

        combined = "\n\n---\n\n".join(parts)

        prompt = f"""Original task: {task}

Results from subtasks:
{combined}

Synthesize these results into a single coherent response. Be concise."""

        try:
            from jarvis.providers import Message

            response = await asyncio.to_thread(
                self.provider.chat,
                messages=[Message(role="user", content=prompt)],
                system="Synthesize multiple results into one coherent response.",
                stream=False,
            )

            if hasattr(response, '__iter__') and not isinstance(response, str):
                response = ''.join(str(c) for c in response)

            return str(response).strip()

        except Exception:
            # Fallback: concatenate results
            return "\n\n".join(r.response for r in results if r.response)

    async def run(self, task: str) -> AsyncGenerator[OrchestratorEvent, None]:
        """
        Run the full orchestration pipeline with progress events.

        Yields OrchestratorEvents for progress tracking.
        """
        start = time.time()

        # Decompose
        subtasks = await self.decompose(task)
        yield OrchestratorEvent(
            type="decompose",
            total_subtasks=len(subtasks),
            result=f"Decomposed into {len(subtasks)} subtask(s)",
        )

        # Execute
        for st in subtasks:
            yield OrchestratorEvent(
                type="subtask_start",
                subtask_id=st.id,
                subtask_description=st.description,
                total_subtasks=len(subtasks),
            )

        results = await self.execute(subtasks)

        for st, result in zip(subtasks, results):
            st.result = result
            yield OrchestratorEvent(
                type="subtask_complete",
                subtask_id=st.id,
                subtask_description=st.description,
                result=result.response[:200],
                total_subtasks=len(subtasks),
            )

        # Merge
        yield OrchestratorEvent(type="merge")
        final = await self.merge_results(task, results)

        duration = time.time() - start
        yield OrchestratorEvent(
            type="done",
            result=final,
            total_subtasks=len(subtasks),
        )
