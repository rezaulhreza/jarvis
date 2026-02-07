"""
Async parallel tool executor with progress events.

Executes multiple tool calls concurrently using asyncio.gather(),
wrapping synchronous tools with asyncio.to_thread().
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, List
from enum import Enum


class EventType(Enum):
    """Types of events emitted during tool execution."""
    TOOL_START = "tool_start"
    TOOL_COMPLETE = "tool_complete"
    TOOL_ERROR = "tool_error"


@dataclass
class ToolResult:
    """Result of a single tool execution."""
    tool_name: str
    args: dict
    result: str
    duration: float
    success: bool
    tool_call_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ToolEvent:
    """Event emitted during tool execution."""
    event_type: EventType
    tool_name: str
    args: dict = field(default_factory=dict)
    result: Optional[str] = None
    duration: Optional[float] = None
    success: bool = True
    tool_call_id: Optional[str] = None
    display: str = ""


class AsyncToolExecutor:
    """
    Executes tool calls in parallel with progress callbacks.
    
    Usage:
        executor = AsyncToolExecutor(tool_map)
        executor.on_tool_start(lambda e: print(f"Starting {e.tool_name}"))
        executor.on_tool_complete(lambda e: print(f"Done {e.tool_name}"))
        results = await executor.execute_parallel(tool_calls)
    """

    def __init__(self, tool_map: dict[str, Callable], max_parallel: int = 5):
        self.tool_map = tool_map
        self.max_parallel = max_parallel
        self._on_start: Optional[Callable] = None
        self._on_complete: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        self._format_display: Optional[Callable] = None

    def on_tool_start(self, callback: Callable[[ToolEvent], Any]):
        """Register callback for tool start events."""
        self._on_start = callback

    def on_tool_complete(self, callback: Callable[[ToolEvent], Any]):
        """Register callback for tool completion events."""
        self._on_complete = callback

    def on_tool_error(self, callback: Callable[[ToolEvent], Any]):
        """Register callback for tool error events."""
        self._on_error = callback

    def set_display_formatter(self, formatter: Callable[[str, dict, Optional[str]], str]):
        """Set function to format tool display text."""
        self._format_display = formatter

    def _get_display(self, tool_name: str, args: dict, result: Optional[str] = None) -> str:
        """Get display text for a tool call."""
        if self._format_display:
            return self._format_display(tool_name, args, result)
        return f"{tool_name}()"

    async def _emit_start(self, tool_name: str, args: dict, tool_call_id: Optional[str] = None):
        """Emit tool start event."""
        if self._on_start:
            event = ToolEvent(
                event_type=EventType.TOOL_START,
                tool_name=tool_name,
                args=args,
                tool_call_id=tool_call_id,
                display=self._get_display(tool_name, args),
            )
            result = self._on_start(event)
            if asyncio.iscoroutine(result):
                await result

    async def _emit_complete(self, tool_result: ToolResult):
        """Emit tool complete event."""
        if self._on_complete:
            event = ToolEvent(
                event_type=EventType.TOOL_COMPLETE,
                tool_name=tool_result.tool_name,
                args=tool_result.args,
                result=tool_result.result,
                duration=tool_result.duration,
                success=tool_result.success,
                tool_call_id=tool_result.tool_call_id,
                display=self._get_display(tool_result.tool_name, tool_result.args, tool_result.result),
            )
            result = self._on_complete(event)
            if asyncio.iscoroutine(result):
                await result

    async def execute_single(
        self,
        tool_name: str,
        args: dict,
        tool_call_id: Optional[str] = None,
    ) -> ToolResult:
        """Execute a single tool call asynchronously."""
        await self._emit_start(tool_name, args, tool_call_id)

        start = time.time()
        try:
            func = self.tool_map.get(tool_name)
            if not func:
                error_msg = f"Unknown tool: {tool_name}"
                result = ToolResult(
                    tool_name=tool_name,
                    args=args,
                    result=error_msg,
                    duration=time.time() - start,
                    success=False,
                    tool_call_id=tool_call_id,
                    error=error_msg,
                )
                await self._emit_complete(result)
                return result

            # Run synchronous tool in thread pool
            output = await asyncio.to_thread(func, **args)
            duration = time.time() - start

            # Check for error results
            success = True
            if isinstance(output, str) and (output.startswith("Error") or output.startswith("error")):
                success = False

            result = ToolResult(
                tool_name=tool_name,
                args=args,
                result=str(output) if output is not None else "",
                duration=duration,
                success=success,
                tool_call_id=tool_call_id,
            )
            await self._emit_complete(result)
            return result

        except Exception as e:
            duration = time.time() - start
            error_msg = f"Error: {e}"
            result = ToolResult(
                tool_name=tool_name,
                args=args,
                result=error_msg,
                duration=duration,
                success=False,
                tool_call_id=tool_call_id,
                error=str(e),
            )
            await self._emit_complete(result)
            return result

    async def execute_parallel(
        self,
        tool_calls: List[dict],
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls in parallel.

        Args:
            tool_calls: List of dicts with keys: tool_name, args, tool_call_id (optional)

        Returns:
            List of ToolResult objects in the same order as input
        """
        if not tool_calls:
            return []

        # If only one tool call, execute directly
        if len(tool_calls) == 1:
            tc = tool_calls[0]
            result = await self.execute_single(
                tc["tool_name"],
                tc.get("args", {}),
                tc.get("tool_call_id"),
            )
            return [result]

        # Use semaphore to limit parallelism
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def _execute_with_semaphore(tc: dict) -> ToolResult:
            async with semaphore:
                return await self.execute_single(
                    tc["tool_name"],
                    tc.get("args", {}),
                    tc.get("tool_call_id"),
                )

        # Execute all in parallel
        results = await asyncio.gather(
            *[_execute_with_semaphore(tc) for tc in tool_calls],
            return_exceptions=False,
        )

        return list(results)

    async def execute_sequential(
        self,
        tool_calls: List[dict],
    ) -> List[ToolResult]:
        """
        Execute tool calls sequentially (when order matters).
        
        Args:
            tool_calls: List of dicts with keys: tool_name, args, tool_call_id (optional)
            
        Returns:
            List of ToolResult objects
        """
        results = []
        for tc in tool_calls:
            result = await self.execute_single(
                tc["tool_name"],
                tc.get("args", {}),
                tc.get("tool_call_id"),
            )
            results.append(result)
        return results
