"""
Tool router - determines which skill to use for a request.

Supports both LLM-based intent classification (preferred) and
keyword matching (fallback) for reliable routing.
"""

import re
from typing import Optional, Dict, Any

from .intent import IntentClassifier, Intent, ClassifiedIntent


# Legacy keyword patterns (used as fallback when intent classification unavailable)
# These are kept for backward compatibility but intent classification is preferred
TOOL_PATTERNS = {
    "get_weather": [
        r"weather\s+(in|for|at)\s+(\w+[\w\s]*)",
        r"(what'?s?|how'?s?|hows)\s+(the\s+)?weather",
        r"temperature\s+(in|for|at)",
        r"forecast\s+(in|for|at)",
    ],
    "web_search": [
        r"search\s+(for|about|the web)",
        r"google\s+",
        r"look\s+up\s+",
        r"find\s+(information|info)\s+(about|on)",
    ],
    "current_time": [
        r"(what|whats|what's)\s+(time|the time)",
        r"(current|right now)\s+time",
        r"time\s+(in|at)\s+\w+",
    ],
    "calculate": [
        r"calculate\s+",
        r"(what|whats|what's)\s+\d+\s*[\+\-\*\/]",
        r"\d+\s*[\+\-\*\/\^]\s*\d+",
        r"(sum|add|multiply|divide|subtract)\s+",
    ],
    "convert_units": [
        r"\d+\s*(kg|lb|km|mi|c|f|celsius|fahrenheit|meters?|feet|inches)",
        r"convert\s+\d+",
    ],
    "read_file": [
        r"(read|show|display|cat|open)\s+(the\s+)?(file|contents)",
        r"(what'?s?|show)\s+(in|inside)\s+",
    ],
    "list_directory": [
        r"(list|show|ls)\s+(files|directory|folder|dir)",
        r"(what'?s?|whats)\s+in\s+(the\s+)?(folder|directory|dir)",
    ],
    "shell_run": [
        r"run\s+(command|cmd)\s+",
        r"execute\s+",
    ],
    "quick_note": [
        r"(save|write|add|make)\s+(a\s+)?(note|reminder)",
        r"(remember|note)\s+(that|this|:)",
    ],
    "github_repos": [
        r"(my|list)\s+(github\s+)?repos",
        r"github\s+repositories",
    ],
}

# Intent to tool mapping
INTENT_TO_TOOL = {
    Intent.WEATHER: "get_weather",
    Intent.TIME_DATE: "current_time",
    Intent.CALCULATE: "calculate",
    Intent.SEARCH: "web_search",
    Intent.NEWS: "get_current_news",
    Intent.FINANCE: "web_search",  # or get_gold_price for gold
    Intent.FILE_OP: "read_file",
    Intent.RECALL: "recall_memory",
    # Multimodal
    Intent.IMAGE_GEN: "generate_image",
    Intent.VIDEO_GEN: "generate_video",
    Intent.MUSIC_GEN: "generate_music",
    Intent.VISION: "analyze_image",
}


def extract_params(user_input: str, tool: str) -> dict:
    """Extract parameters from user input based on tool type."""
    input_lower = user_input.lower()
    params = {}

    if tool == "get_weather":
        # Extract city name
        match = re.search(r"weather\s+(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\s+now|\s+today|\s*\?|$)", input_lower)
        if match:
            params["city"] = match.group(1).strip().title()
        else:
            # Try to find any capitalized words as city
            words = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', user_input)
            if words:
                params["city"] = words[-1]

    elif tool == "web_search":
        # Everything after "search for" or similar
        match = re.search(r"(?:search|look up|find|google)\s+(?:for\s+|about\s+)?(.+)", input_lower)
        if match:
            params["query"] = match.group(1).strip()
        else:
            params["query"] = user_input

    elif tool == "current_time":
        # Extract timezone if mentioned
        match = re.search(r"time\s+(?:in|at)\s+([a-zA-Z_/]+)", input_lower)
        if match:
            params["timezone"] = match.group(1)

    elif tool == "calculate":
        # Extract expression
        match = re.search(r"(?:calculate|compute|what'?s?)\s+(.+)", input_lower)
        if match:
            params["expression"] = match.group(1).strip()
        else:
            # Look for math expression
            match = re.search(r'(\d+[\s\d\+\-\*\/\^\(\)\.]+)', user_input)
            if match:
                params["expression"] = match.group(1).strip()

    elif tool == "read_file":
        # Extract file path
        match = re.search(r'([/~][\w/\.\-]+)', user_input)
        if match:
            params["path"] = match.group(1)

    elif tool == "list_directory":
        # Extract directory path
        match = re.search(r'([/~][\w/\.\-]+)', user_input)
        if match:
            params["path"] = match.group(1)
        else:
            params["path"] = "."

    elif tool == "quick_note":
        # Extract note content
        match = re.search(r"(?:note|remember|save)(?:\s+that)?\s*:?\s*(.+)", input_lower)
        if match:
            params["content"] = match.group(1).strip()
        else:
            params["content"] = user_input

    elif tool == "generate_image":
        # Extract image prompt - everything after generation keywords
        match = re.search(r"(?:draw|create|generate|make|paint|sketch|design)\s+(?:an?\s+)?(?:image\s+(?:of\s+)?)?(.+)", input_lower)
        if match:
            params["prompt"] = match.group(1).strip()
        else:
            params["prompt"] = user_input

    elif tool == "generate_video":
        # Extract video prompt
        match = re.search(r"(?:create|generate|make|animate)\s+(?:a\s+)?(?:video\s+(?:of\s+)?)?(.+)", input_lower)
        if match:
            params["prompt"] = match.group(1).strip()
        else:
            params["prompt"] = user_input

    elif tool == "generate_music":
        # Extract music prompt
        match = re.search(r"(?:create|generate|make|compose)\s+(?:a\s+)?(?:music|song|soundtrack|jingle)\s+(?:for\s+|about\s+)?(.+)", input_lower)
        if match:
            params["prompt"] = match.group(1).strip()
        else:
            params["prompt"] = user_input

    elif tool == "analyze_image":
        # Extract image path and prompt
        path_match = re.search(r'([/~][\w/\.\-]+\.(jpg|jpeg|png|gif|webp))', user_input, re.IGNORECASE)
        if path_match:
            params["image_path"] = path_match.group(1)
        # The prompt is the whole question
        params["prompt"] = user_input

    return params


class ToolRouter:
    """
    Routes user requests to appropriate tools.

    Supports LLM-based intent classification (preferred) with
    keyword pattern fallback for reliability.
    """

    def __init__(self, ollama_client=None, router_model: str = "functiongemma", config: dict = None):
        self.client = ollama_client
        self.router_model = router_model
        self.config = config or {}

        # Initialize intent classifier if config enables it
        self.classifier = None
        if self.config.get("intent", {}).get("enabled", True):
            self.classifier = IntentClassifier(provider=ollama_client, config=config)

    def route(self, user_input: str, context: dict = None) -> dict:
        """
        Determine which tool to use.

        Priority:
        1. Intent classification (if enabled and confident)
        2. Keyword pattern matching (fast, reliable)
        3. LLM routing (for complex cases)
        """
        input_lower = user_input.lower()

        # === Try intent classification first ===
        if self.classifier:
            try:
                intent = self.classifier.classify_sync(user_input)
                confidence_threshold = self.config.get("intent", {}).get("confidence_threshold", 0.7)

                if intent.confidence >= confidence_threshold:
                    tool = self._intent_to_tool(intent, user_input)
                    if tool:
                        params = extract_params(user_input, tool)
                        return {
                            "tool": tool,
                            "params": params,
                            "method": "intent",
                            "intent": intent.intent.value,
                            "confidence": intent.confidence,
                            "reasoning_level": intent.reasoning_level.value,
                        }
            except Exception as e:
                print(f"[ToolRouter] Intent classification failed: {e}")

        # === Fall back to keyword patterns ===
        for tool, patterns in TOOL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    params = extract_params(user_input, tool)
                    return {
                        "tool": tool,
                        "params": params,
                        "method": "pattern"
                    }

        # Check for simple greetings - no tool needed
        if self._is_greeting(input_lower):
            return {"tool": "none", "params": {}, "method": "greeting"}

        # For complex requests, use LLM router if available
        if self.client and self._is_complex_request(user_input):
            return self._llm_route(user_input, context)

        # Default: no tool, just chat
        return {"tool": "none", "params": {}}

    def _intent_to_tool(self, intent: ClassifiedIntent, user_input: str) -> Optional[str]:
        """Convert classified intent to tool name."""
        if not intent.requires_tools:
            return None

        # Direct mapping
        if intent.intent in INTENT_TO_TOOL:
            tool = INTENT_TO_TOOL[intent.intent]

            # Special case: Finance with gold -> get_gold_price
            if intent.intent == Intent.FINANCE and "gold" in user_input.lower():
                import os
                if os.getenv("GOLDAPI_KEY") or os.getenv("GOLD_API_KEY"):
                    return "get_gold_price"

            return tool

        # Use suggested tools from classifier
        if intent.suggested_tools:
            return intent.suggested_tools[0]

        return None

    def _is_greeting(self, text: str) -> bool:
        """Check if text is a simple greeting."""
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good evening', 'good afternoon',
                     'how are you', 'what\'s up', 'howdy', 'greetings']
        return any(text.strip() == g or text.startswith(g + " ") or text.startswith(g + ",")
                   for g in greetings)

    def _is_complex_request(self, user_input: str) -> bool:
        """Check if request needs LLM routing."""
        # Long requests or questions might need routing
        return len(user_input) > 50 or '?' in user_input

    def _llm_route(self, user_input: str, context: dict) -> dict:
        """Use LLM to determine tool (slower but handles edge cases)."""
        try:
            from ..skills import get_skills_schema

            prompt = f"""You are a tool router. Given a user request, output JSON for which tool to use.

Available tools:
{get_skills_schema()}

User request: {user_input}

Respond with ONLY JSON: {{"tool": "tool_name", "params": {{...}}}}
If no tool needed, respond: {{"tool": "none", "params": {{}}}}"""

            response = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.router_model,
                stream=False
            )

            return self._parse_json(response)

        except Exception:
            return {"tool": "none", "params": {}}

    def _parse_json(self, response: str) -> dict:
        """Extract JSON from response."""
        import json
        try:
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        return {"tool": "none", "params": {}}


def should_use_reasoning(user_input: str, classifier: IntentClassifier = None) -> bool:
    """
    Check if deep reasoning is needed.

    Uses intent classification if available, otherwise falls back to keywords.
    """
    if classifier:
        try:
            intent = classifier.classify_sync(user_input)
            from .intent import ReasoningLevel
            return intent.reasoning_level == ReasoningLevel.DEEP
        except Exception:
            pass

    # Fallback to keywords
    keywords = ['why', 'explain', 'analyze', 'compare', 'debug', 'figure out',
                'step by step', 'in detail', 'thoroughly', 'comprehensive']
    return any(kw in user_input.lower() for kw in keywords)


def should_use_vision(user_input: str, classifier: IntentClassifier = None, has_image: bool = False) -> bool:
    """
    Check if vision model is needed.

    Uses intent classification if available, otherwise falls back to patterns.

    Args:
        user_input: User's message
        classifier: Intent classifier instance
        has_image: Whether an image attachment is present
    """
    if classifier:
        try:
            intent = classifier.classify_sync(user_input)
            return intent.intent == Intent.VISION
        except Exception:
            pass

    # If image is attached, likely needs vision
    if has_image:
        return True

    # Fallback to patterns
    patterns = [
        r'\.(jpg|jpeg|png|gif|webp|bmp)\b',
        r'(this|the)\s+(image|picture|photo|screenshot)',
        r'analyze\s+(this|the)\s+image',
        r'what\'?s?\s+in\s+(this|the)\s+(image|picture|photo)',
    ]
    input_lower = user_input.lower()
    return any(re.search(p, input_lower) for p in patterns)


def should_generate_image(user_input: str, classifier: IntentClassifier = None) -> bool:
    """Check if image generation is needed."""
    if classifier:
        try:
            intent = classifier.classify_sync(user_input)
            return intent.intent == Intent.IMAGE_GEN
        except Exception:
            pass

    # Fallback patterns
    patterns = [
        r'\bdraw\b', r'\bsketch\b', r'\bpaint\b',
        r'\bcreate\s+(an?\s+)?image\b', r'\bgenerate\s+(an?\s+)?image\b',
        r'\bmake\s+(an?\s+)?(image|picture|art)\b',
    ]
    input_lower = user_input.lower()
    return any(re.search(p, input_lower) for p in patterns)


def should_generate_video(user_input: str, classifier: IntentClassifier = None) -> bool:
    """Check if video generation is needed."""
    if classifier:
        try:
            intent = classifier.classify_sync(user_input)
            return intent.intent == Intent.VIDEO_GEN
        except Exception:
            pass

    patterns = [
        r'\bcreate\s+(a\s+)?video\b', r'\bgenerate\s+(a\s+)?video\b',
        r'\bmake\s+(a\s+)?video\b', r'\banimate\b',
    ]
    input_lower = user_input.lower()
    return any(re.search(p, input_lower) for p in patterns)


def should_generate_music(user_input: str, classifier: IntentClassifier = None) -> bool:
    """Check if music generation is needed."""
    if classifier:
        try:
            intent = classifier.classify_sync(user_input)
            return intent.intent == Intent.MUSIC_GEN
        except Exception:
            pass

    patterns = [
        r'\bcreate\s+(a\s+)?music\b', r'\bgenerate\s+(a\s+)?music\b',
        r'\bmake\s+(a\s+)?song\b', r'\bcompose\b',
    ]
    input_lower = user_input.lower()
    return any(re.search(p, input_lower) for p in patterns)


def get_reasoning_level(user_input: str, classifier: IntentClassifier = None) -> str:
    """
    Get the recommended reasoning level for a user input.

    Returns: "fast", "balanced", or "deep"
    """
    if classifier:
        try:
            intent = classifier.classify_sync(user_input)
            return intent.reasoning_level.value
        except Exception:
            pass

    # Fallback heuristics
    input_lower = user_input.lower()

    # Fast indicators
    fast_patterns = [r'^(hi|hello|hey|thanks|ok|yes|no)\b', r'^\w+\?$']
    for p in fast_patterns:
        if re.search(p, input_lower):
            return "fast"

    # Deep indicators
    deep_keywords = ['explain', 'analyze', 'compare', 'debug', 'step by step',
                     'in detail', 'thoroughly', 'comprehensive', 'why']
    if any(kw in input_lower for kw in deep_keywords):
        return "deep"

    return "balanced"
