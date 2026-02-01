"""Tool router using FunctionGemma or similar model"""

import json
import re
from typing import Optional


# Tool definitions for the router
TOOLS_SCHEMA = """
Available tools:
1. web_search(query: str) - Search the web for information
2. shell_run(command: str) - Run a shell command (safe commands only)
3. read_file(path: str) - Read contents of a file
4. list_directory(path: str) - List files in a directory
5. save_fact(fact: str) - Remember a fact about the user
6. get_facts() - Recall facts about the user
7. analyze_image(path: str, question: str) - Analyze an image
8. think(problem: str) - Use deep reasoning for complex problems
9. none() - No tool needed, just respond conversationally

Respond with JSON: {"tool": "tool_name", "params": {...}, "reasoning": "why this tool"}
"""


class ToolRouter:
    """Routes user requests to appropriate tools using a fast model."""

    def __init__(self, ollama_client, router_model: str = "functiongemma"):
        self.client = ollama_client
        self.router_model = router_model

    def route(self, user_input: str, context: dict = None) -> dict:
        """
        Determine which tool to use for a given input.

        Args:
            user_input: The user's message
            context: Optional context (working memory, etc.)

        Returns:
            Dict with tool name and parameters
        """
        prompt = f"""{TOOLS_SCHEMA}

User request: {user_input}

Context: {json.dumps(context) if context else 'None'}

What tool should be used? Respond with JSON only."""

        try:
            response = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.router_model,
                stream=False
            )

            # Parse JSON from response
            return self._parse_response(response)

        except Exception as e:
            # Default to no tool on error
            return {"tool": "none", "params": {}, "error": str(e)}

    def _parse_response(self, response: str) -> dict:
        """Extract JSON from model response."""
        # Try to find JSON in the response
        try:
            # Look for JSON pattern
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # If parsing fails, try to infer from keywords
        response_lower = response.lower()

        if 'search' in response_lower or 'web' in response_lower:
            return {"tool": "web_search", "params": {"query": ""}, "inferred": True}
        elif 'file' in response_lower or 'read' in response_lower:
            return {"tool": "read_file", "params": {"path": ""}, "inferred": True}
        elif 'image' in response_lower or 'picture' in response_lower:
            return {"tool": "analyze_image", "params": {}, "inferred": True}

        return {"tool": "none", "params": {}}


def should_use_reasoning(user_input: str) -> bool:
    """Heuristic to decide if deep reasoning is needed."""
    reasoning_keywords = [
        'why', 'explain', 'analyze', 'compare', 'plan',
        'strategy', 'think through', 'reason', 'debug',
        'figure out', 'solve', 'complex', 'tricky'
    ]

    input_lower = user_input.lower()
    return any(kw in input_lower for kw in reasoning_keywords)


def should_use_vision(user_input: str) -> bool:
    """Check if vision model is needed."""
    vision_keywords = [
        'image', 'picture', 'photo', 'screenshot', 'see',
        'look at', 'what is this', 'describe this', '.jpg',
        '.png', '.jpeg', '.gif', '.webp'
    ]

    input_lower = user_input.lower()
    return any(kw in input_lower for kw in vision_keywords)
