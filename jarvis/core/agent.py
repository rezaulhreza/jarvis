"""
Agentic loop with tool calling.

Supports both native tool calling and prompt-based fallback.
Integrates RAG for knowledge retrieval.
Uses LLM-based intent classification for intelligent routing.
"""

import re
import json
from typing import List, Optional, Tuple
from pathlib import Path

from .tools import (
    read_file, list_files, search_files, run_command,
    write_file, edit_file, get_project_structure, set_project_root, set_ui,
    clear_read_files, ALL_TOOLS,
    # File operations
    glob_files, grep,
    # Git operations
    git_status, git_diff, git_log, git_commit, git_add, git_branch, git_stash,
    # Web
    web_search, web_fetch, get_current_news, get_gold_price, get_weather, get_current_time,
    # Utilities
    calculate, save_memory, recall_memory, github_search,
    # Task management
    task_create, task_update, task_list, task_get,
)
# Media generation (lazy import to avoid startup delays)
def _get_media_tools():
    """Lazy import of media tools."""
    from jarvis.skills.media_gen import generate_image, generate_video, generate_music, analyze_image
    return generate_image, generate_video, generate_music, analyze_image
from .intent import IntentClassifier, Intent, ReasoningLevel, ClassifiedIntent

class Agent:
    """Agent with tool calling - native or prompt-based."""

    def __init__(self, provider, project_root: Path, ui=None, config: dict | None = None):
        self.provider = provider
        self.project_root = project_root
        self.ui = ui
        self.config = config or {}
        self.max_iterations = 15
        self.last_streamed = False
        set_project_root(project_root)
        set_ui(ui)  # Pass UI to tools for confirmations

        # Initialize intent classifier
        self.classifier = IntentClassifier(provider=provider, config=config)
        self._last_intent: Optional[ClassifiedIntent] = None
        self._user_reasoning_level: Optional[str] = None  # User override from CLI

    def _supports_native_tools(self) -> bool:
        """Check if current model supports native tool calling."""
        agent_cfg = self.config.get("agent", {})
        tool_mode = (agent_cfg.get("tool_mode", "auto") or "auto").lower()

        if tool_mode == "off":
            return False
        if tool_mode == "prompt":
            return False
        if tool_mode == "native":
            return bool(getattr(self.provider, "supports_tools", False))
        # auto
        return bool(getattr(self.provider, "supports_tools", False))

    def _get_timeout(self) -> int:
        """Get appropriate timeout based on model type."""
        agent_cfg = self.config.get("agent", {})
        timeouts = agent_cfg.get("timeouts", {})
        default_timeout = int(timeouts.get("default", 120))
        reasoning_timeout = int(timeouts.get("reasoning", 300))
        reasoning_model = (self.config.get("models", {}) or {}).get("reasoning")

        if reasoning_model and reasoning_model in (self.provider.model or ""):
            return reasoning_timeout
        return default_timeout

    def _get_timeout_for_level(self, level: ReasoningLevel) -> int:
        """Get timeout based on reasoning level."""
        reasoning_cfg = self.config.get("reasoning", {}).get("levels", {})
        level_cfg = reasoning_cfg.get(level.value, {})
        return int(level_cfg.get("timeout", self._get_timeout()))

    def classify_intent(self, message: str, context: List = None) -> ClassifiedIntent:
        """
        Classify user message intent using the IntentClassifier.

        This is the primary entry point for intent-based routing.
        """
        self._last_intent = self.classifier.classify_sync(message, context)
        return self._last_intent

    def get_reasoning_level(self, intent: ClassifiedIntent = None, user_override: str = None) -> ReasoningLevel:
        """
        Determine reasoning level from intent or user override.

        Args:
            intent: Classified intent (uses last classified if None)
            user_override: User-specified level ("fast", "balanced", "deep")

        Returns:
            ReasoningLevel enum value
        """
        # Check user override (explicit parameter or instance-level setting)
        override = user_override or self._user_reasoning_level
        if override:
            try:
                return ReasoningLevel(override.lower())
            except ValueError:
                pass

        intent = intent or self._last_intent
        if intent:
            return intent.reasoning_level

        # Default from config
        default = self.config.get("reasoning", {}).get("default_level", "balanced")
        try:
            return ReasoningLevel(default)
        except ValueError:
            return ReasoningLevel.BALANCED

    def get_model_for_level(self, level: ReasoningLevel) -> str:
        """
        Select appropriate model for reasoning level.

        Maps reasoning levels to configured models.
        """
        reasoning_cfg = self.config.get("reasoning", {}).get("levels", {})
        level_cfg = reasoning_cfg.get(level.value, {})
        model_type = level_cfg.get("model", "default")

        # Get actual model from provider or config
        if hasattr(self.provider, "get_model_for_task"):
            return self.provider.get_model_for_task(model_type)

        # Fallback to models config
        models = self.config.get("models", {})
        if model_type in models:
            return models[model_type]

        return self.provider.model

    def detect_tool_from_intent(self, intent: ClassifiedIntent, message: str) -> Optional[Tuple[str, dict]]:
        """
        Detect which tool to run based on classified intent.

        This replaces keyword-based _detect_auto_tool with intent-based routing.
        """
        if not intent.requires_tools:
            return None

        msg = message.strip()
        msg_lower = msg.lower()

        # Map intents to tool detection
        if intent.intent == Intent.WEATHER:
            match = re.search(r'weather\s+(?:in|for)\s+(.+)', msg_lower)
            if match:
                city = match.group(1).strip().title()
                return "get_weather", {"city": city}
            return None  # Let model ask for location

        elif intent.intent == Intent.TIME_DATE:
            # Only trigger if explicitly asking about time (not incidental mentions)
            time_patterns = [
                r"what\s*(?:'s|is)?\s*(?:the\s+)?time\b",
                r"current\s+time\b",
                r"time\s+(?:now|right\s+now)\b",
                r"time\s+(?:in|for)\s+\w+",
                r"what\s+time\s+(?:in|is\s+it)",
                r"tell\s+me\s+(?:the\s+)?time\b",
            ]
            if not any(re.search(p, msg_lower) for p in time_patterns):
                return None  # Not an explicit time request
            match = re.search(r'time\s+(?:in|for)\s+(.+)', msg_lower)
            if match:
                tz = match.group(1).strip().upper()
                return "get_current_time", {"timezone": tz}
            return "get_current_time", {"timezone": "UTC"}

        elif intent.intent == Intent.CALCULATE:
            # Extract expression
            math_match = re.search(r'(?:calc(?:ulate)?\s+)?([0-9\.\+\-\*\/\(\)\s]+)', msg_lower)
            if math_match:
                expr = math_match.group(1).strip()
                if expr and re.match(r'^[\d\.\+\-\*\/\(\)\s]+$', expr):
                    return "calculate", {"expression": expr}

        elif intent.intent == Intent.NEWS:
            topic = msg_lower.replace("news", "").replace("headlines", "").strip()
            return "get_current_news", {"topic": topic or msg}

        elif intent.intent == Intent.FINANCE:
            # Check for gold price with API key
            if "gold" in msg_lower:
                import os
                if os.getenv("GOLDAPI_KEY") or os.getenv("GOLD_API_KEY"):
                    currency = "USD"
                    for cur in ["GBP", "EUR", "USD", "AED", "AUD", "CAD", "CHF", "JPY"]:
                        if cur.lower() in msg_lower:
                            currency = cur
                            break
                    return "get_gold_price", {"currency": currency}
            # Fallback to web search for other financial queries
            return "web_search", {"query": msg}

        elif intent.intent == Intent.SEARCH:
            # Extract search query
            for prefix in ["search for ", "look up ", "find online ", "google ", "search the web for "]:
                if msg_lower.startswith(prefix):
                    topic = msg[len(prefix):].strip()
                    if topic:
                        return "web_search", {"query": topic}
            return "web_search", {"query": msg}

        elif intent.intent == Intent.RECALL:
            return "recall_memory", {"query": msg}

        # For file_op, git, shell, code - don't auto-run, let the full agent handle
        return None

    def requires_full_agent_from_intent(self, intent: ClassifiedIntent) -> bool:
        """
        Check if intent requires full agent mode (file ops, git, shell).

        This replaces the keyword-based _requires_full_agent method.
        """
        full_agent_intents = {Intent.FILE_OP, Intent.GIT, Intent.SHELL, Intent.CODE}
        return intent.intent in full_agent_intents

    def _requires_full_agent(self, user_message: str) -> bool:
        """Check if this query requires full agent capabilities (file ops, git, etc.)."""
        if not user_message:
            return False

        msg = user_message.lower()

        # File/code operations that need agent mode
        file_ops = [
            "read file", "open file", "show file", "cat ", "edit ", "modify ",
            "write ", "create file", "save to", "update file", "change file",
            "refactor", "fix bug", "debug", "traceback", "stack trace",
            "in the file", "in file", "this file", "that file"
        ]
        if any(op in msg for op in file_ops):
            return True

        # Git operations
        git_ops = [
            "git ", "commit", "push", "pull", "branch", "merge", "rebase",
            "stash", "checkout", "git diff", "git status", "git log"
        ]
        if any(op in msg for op in git_ops):
            return True

        # Shell commands
        shell_ops = ["run command", "execute", "npm ", "pip ", "python ", "node "]
        if any(op in msg for op in shell_ops):
            return True

        # Explicit file paths
        import re
        if re.search(r'\.(py|js|ts|tsx|jsx|go|rs|java|cpp|c|rb|php|yaml|yml|json|md|txt)\b', msg):
            return True

        return False

    def _likely_needs_tools(self, user_message: str) -> bool:
        """Heuristic to decide if tools are likely needed for this request."""
        if not user_message:
            return False

        msg = user_message.lower()
        explicit_search = [
            "web search", "search web", "search the web", "google", "look up",
            "find online", "search for", "browse"
        ]
        if any(s in msg for s in explicit_search):
            return True

        # Current info / web lookups - things that change and need real-time data
        current_signals = [
            "current", "latest", "today", "right now", "recent", "breaking",
            "news", "headline", "president", "prime minister", "ceo", "stock",
            "price", "score", "election", "as of", "this year", "this month",
            "2024", "2025", "2026", "2027",
            # Financial
            "bitcoin", "crypto", "market", "trading", "forex", "gold", "silver",
            "dollar", "euro", "pound", "exchange rate",
            # Events
            "match", "game", "tournament", "championship", "world cup",
            # Tech
            "release", "version", "announced", "launched"
        ]
        if any(s in msg for s in current_signals):
            return True

        # Question patterns likely needing lookup
        lookup_patterns = [
            r"who is (the|a|an)?\s*\w+",  # Who is X
            r"what is the\s+\w+\s+(price|rate|score|status)",  # What is the X price
            r"how much (is|does|are|cost)",  # How much is/does
            r"where (is|are|can)",  # Where is/are/can
            r"when (is|does|did|will)",  # When is/does/did/will
        ]
        for pattern in lookup_patterns:
            if re.search(pattern, msg):
                return True

        # Local code / file operations
        file_signals = [
            "file", "repo", "project", "codebase", "function", "class",
            "line", "stack trace", "traceback", "error", "bug", "fix",
            "refactor", "edit", "update", "read", "open", "search",
            "commit", "branch", "git", "push", "pull", "merge"
        ]
        if any(s in msg for s in file_signals):
            return True

        # File extensions hinting code questions
        if re.search(r"\.(py|js|ts|tsx|jsx|go|rs|java|cpp|c|rb|php|yaml|yml|json)\b", msg):
            return True

        return False

    def _detect_auto_tool(self, user_message: str) -> Optional[Tuple[str, dict]]:
        """Detect when we should auto-run a tool instead of relying on tool calling."""
        agent_cfg = self.config.get("agent", {})
        if not agent_cfg.get("auto_tools", True):
            return None

        msg = (user_message or "").strip()
        msg_lower = msg.lower()

        # Weather
        if agent_cfg.get("auto_weather", True) and "weather" in msg_lower:
            match = re.search(r"weather\s+(?:in|for)\s+(.+)", msg_lower)
            if match:
                city = match.group(1).strip().title()
                return "get_weather", {"city": city}
            # Fallback to web search if location not obvious
            return "web_search", {"query": msg}

        # Time - only match explicit time queries, not incidental mentions
        time_patterns = [
            r"what\s*(?:'s|is)?\s*(?:the\s+)?time\b",  # what time, what's the time
            r"current\s+time\b",  # current time
            r"time\s+(?:now|right\s+now)\b",  # time now
            r"time\s+(?:in|for)\s+\w+",  # time in <location>
            r"what\s+time\s+(?:in|is\s+it\s+in)",  # what time in <location>
            r"tell\s+me\s+(?:the\s+)?time\b",  # tell me the time
        ]
        if agent_cfg.get("auto_time", True) and any(re.search(p, msg_lower) for p in time_patterns):
            match = re.search(r"time\s+(?:in|for)\s+(.+)", msg_lower)
            if match:
                tz = match.group(1).strip().upper()
                return "get_current_time", {"timezone": tz}
            return "get_current_time", {"timezone": "UTC"}

        # Simple calculations
        if agent_cfg.get("auto_calculate", True):
            math_match = re.search(r"^(?:calc(?:ulate)?\s+)?([0-9\.\+\-\*\/\(\)\s]+)$", msg_lower)
            if math_match:
                expr = math_match.group(1).strip()
                if expr:
                    return "calculate", {"expression": expr}

        # Current info / news / real-time data
        if agent_cfg.get("auto_current_info", True):
            # Explicit web search intent - but skip command-only phrases
            # These are handled separately in the UI to use conversation context
            command_only = [
                "use web search", "do a web search", "search the web", "search web",
                "use search", "web search it", "google it", "look it up", "search online"
            ]
            if msg_lower.strip() in command_only:
                return None  # Let UI handle with conversation context

            # Search WITH a topic
            explicit_search = ["search for ", "look up ", "find online ", "google "]
            for prefix in explicit_search:
                if msg_lower.startswith(prefix):
                    topic = msg[len(prefix):].strip()
                    if topic:
                        return "web_search", {"query": topic}
            # Weather queries: prefer get_weather when location is provided
            if "weather" in msg_lower:
                city_match = re.search(r"\b(?:in|for)\s+([a-zA-Z\s\-]+)$", msg_lower)
                if city_match:
                    city = city_match.group(1).strip().title()
                    if city:
                        return "get_weather", {"city": city}
                # No location provided; avoid web_search so the model asks a clarifying question
                return None

            # Explicit GoldAPI requests
            if "goldapi" in msg_lower or "gold api" in msg_lower:
                import os
                if os.getenv("GOLDAPI_KEY") or os.getenv("GOLD_API_KEY"):
                    currency = "USD"
                    for cur in ["GBP", "EUR", "USD", "AED", "AUD", "CAD", "CHF", "JPY"]:
                        if cur.lower() in msg_lower:
                            currency = cur
                            break
                    return "get_gold_price", {"currency": currency}

            # Explicit current/latest signals
            current_signals = [
                "current", "latest", "today", "right now", "recent", "breaking",
                "news", "headline", "as of", "this week", "this month", "this year",
                "2024", "2025", "2026", "2027"  # Recent/future years
            ]
            if any(s in msg_lower for s in current_signals):
                if "news" in msg_lower or "headline" in msg_lower:
                    topic = msg_lower.replace("news", "").replace("headlines", "").strip()
                    return "get_current_news", {"topic": topic or msg}
                return "web_search", {"query": msg}

            # Leadership and political figures (changes frequently)
            leadership_signals = [
                "president", "prime minister", "ceo", "chairman", "chancellor",
                "governor", "minister", "secretary", "mayor", "leader"
            ]
            if any(s in msg_lower for s in leadership_signals):
                return "web_search", {"query": msg}

            # Financial/market data (always needs real-time)
            financial_signals = [
                "price", "stock", "share", "market", "bitcoin", "crypto", "btc", "eth",
                "gold", "silver", "oil", "forex", "exchange rate", "dollar", "euro",
                "pound", "yen", "trading", "nasdaq", "dow", "s&p", "ftse", "index"
            ]
            if any(s in msg_lower for s in financial_signals):
                # Prefer GoldAPI for gold price if configured
                if "gold" in msg_lower:
                    import os
                    if os.getenv("GOLDAPI_KEY") or os.getenv("GOLD_API_KEY"):
                        currency = "USD"
                        for cur in ["GBP", "EUR", "USD", "AED", "AUD", "CAD", "CHF", "JPY"]:
                            if cur.lower() in msg_lower:
                                currency = cur
                                break
                        return "get_gold_price", {"currency": currency}
                return "web_search", {"query": msg}

            # Sports and events
            sports_signals = [
                "score", "match", "game", "won", "lost", "playing", "tournament",
                "championship", "league", "world cup", "olympics", "super bowl"
            ]
            if any(s in msg_lower for s in sports_signals):
                return "web_search", {"query": msg}

            # Tech/product releases
            tech_signals = [
                "release", "launched", "announced", "version", "update", "patch",
                "iphone", "android", "windows", "macos", "ios"
            ]
            if any(s in msg_lower for s in tech_signals) and any(w in msg_lower for w in ["new", "latest", "when", "what"]):
                return "web_search", {"query": msg}

            # Question patterns that likely need current info
            question_patterns = [
                r"who is the .*(president|ceo|leader|minister|head)",
                r"what is the .*(price|rate|score|status|situation)",
                r"how much (is|does|are).*cost",
                r"where (is|are).*happening",
                r"when (is|does|will).*\?",
                r"is .*(open|closed|available|happening)",
            ]
            for pattern in question_patterns:
                if re.search(pattern, msg_lower):
                    if "gold" in msg_lower:
                        import os
                        if os.getenv("GOLDAPI_KEY") or os.getenv("GOLD_API_KEY"):
                            currency = "USD"
                            for cur in ["GBP", "EUR", "USD", "AED", "AUD", "CAD", "CHF", "JPY"]:
                                if cur.lower() in msg_lower:
                                    currency = cur
                                    break
                            return "get_gold_price", {"currency": currency}
                    if "weather" in msg_lower:
                        return None
                    return "web_search", {"query": msg}

        # === MULTIMODAL GENERATION ===
        # Image generation
        image_gen_patterns = [
            r'\bdraw\b', r'\bsketch\b', r'\bpaint\b',
            r'\bcreate\s+(an?\s+)?image\b', r'\bgenerate\s+(an?\s+)?image\b',
            r'\bmake\s+(an?\s+)?(image|picture|art)\b', r'\billustrat',
        ]
        for pattern in image_gen_patterns:
            if re.search(pattern, msg_lower):
                # Extract prompt
                match = re.search(r"(?:draw|create|generate|make|paint|sketch|design)\s+(?:an?\s+)?(?:image\s+(?:of\s+)?)?(.+)", msg_lower)
                prompt = match.group(1).strip() if match else msg
                return "generate_image", {"prompt": prompt}

        # Video generation - expanded patterns
        video_gen_patterns = [
            r'\bcreate\s+(a\s+)?video\b', r'\bgenerate\s+(a\s+)?video\b',
            r'\bmake\s+(a\s+)?video\b', r'\banimate\b', r'\bvideo\s+of\b',
            r'\bclip\s+of\b', r'\banimation\s+of\b', r'\bshort\s+video\b',
            r'\bturn\s+(this\s+)?into\s+(a\s+)?video\b',
            r'\bi\s+want\s+(a\s+)?video\b', r'\bneed\s+(a\s+)?video\b',
        ]
        for pattern in video_gen_patterns:
            if re.search(pattern, msg_lower):
                # Extract the actual content prompt
                match = re.search(r"(?:create|generate|make|animate|video of|clip of|animation of)\s*(?:a\s+)?(?:video\s+(?:of\s+)?)?(.+)", msg_lower)
                prompt = match.group(1).strip() if match else msg
                # Clean up common prefixes
                prompt = re.sub(r'^(?:a\s+)?(?:video\s+)?(?:of\s+)?', '', prompt).strip()
                if not prompt:
                    prompt = msg
                return "generate_video", {"prompt": prompt}

        # Music generation
        music_gen_patterns = [
            r'\bcreate\s+(a\s+)?music\b', r'\bgenerate\s+(a\s+)?music\b',
            r'\bmake\s+(a\s+)?song\b', r'\bcompose\b', r'\bmusic\s+for\b',
        ]
        for pattern in music_gen_patterns:
            if re.search(pattern, msg_lower):
                match = re.search(r"(?:create|generate|make|compose)\s+(?:a\s+)?(?:music|song|soundtrack|jingle)\s+(?:for\s+|about\s+)?(.+)", msg_lower)
                prompt = match.group(1).strip() if match else msg
                return "generate_music", {"prompt": prompt}

        return None

    def detect_auto_tool(self, message: str, use_intent: bool = True) -> Optional[Tuple[str, dict]]:
        """
        Unified tool detection - uses intent classification when enabled.

        This is the preferred entry point for auto-tool detection.
        Falls back to keyword-based detection if intent classification fails.

        Args:
            message: User message
            use_intent: Whether to try intent-based detection first

        Returns:
            Tuple of (tool_name, args) or None
        """
        intent_cfg = self.config.get("intent", {})

        # Try intent-based detection if enabled
        if use_intent and intent_cfg.get("enabled", True):
            try:
                intent = self.classify_intent(message)
                if intent.confidence >= intent_cfg.get("confidence_threshold", 0.7):
                    tool = self.detect_tool_from_intent(intent, message)
                    if tool:
                        return tool
            except Exception as e:
                print(f"[Agent] Intent classification failed: {e}")

        # Fall back to keyword-based detection
        return self._detect_auto_tool(message)

    def likely_needs_tools(self, message: str, use_intent: bool = True) -> bool:
        """
        Check if message likely needs tools - uses intent classification when enabled.

        Args:
            message: User message
            use_intent: Whether to use intent-based detection

        Returns:
            True if tools are likely needed
        """
        intent_cfg = self.config.get("intent", {})

        if use_intent and intent_cfg.get("enabled", True):
            try:
                intent = self.classify_intent(message)
                if intent.confidence >= intent_cfg.get("confidence_threshold", 0.7):
                    return intent.requires_tools
            except Exception:
                pass

        return self._likely_needs_tools(message)

    def requires_full_agent(self, message: str, use_intent: bool = True) -> bool:
        """
        Check if message requires full agent - uses intent classification when enabled.

        Args:
            message: User message
            use_intent: Whether to use intent-based detection

        Returns:
            True if full agent mode is needed
        """
        intent_cfg = self.config.get("intent", {})

        if use_intent and intent_cfg.get("enabled", True):
            try:
                intent = self.classify_intent(message)
                if intent.confidence >= intent_cfg.get("confidence_threshold", 0.7):
                    return self.requires_full_agent_from_intent(intent)
            except Exception:
                pass

        return self._requires_full_agent(message)

    def _get_tools_prompt(self) -> str:
        """Get prompt describing available tools for non-native models."""
        return '''
You have access to tools. When you need to use a tool, output ONLY a valid JSON object.

IMPORTANT FORMATTING RULES:
1. Output ONLY the JSON object, nothing before or after
2. Do NOT wrap in markdown code blocks (no ```json)
3. Use exact parameter names shown in examples
4. All string values must be in double quotes

EXAMPLE - Read a file:
{"tool": "read_file", "path": "src/main.py"}

EXAMPLE - Search the web:
{"tool": "web_search", "query": "current gold price USD"}

EXAMPLE - Get weather:
{"tool": "get_weather", "city": "London"}

AVAILABLE TOOLS:

FILES:
  read_file       - {"tool": "read_file", "path": "path/to/file"}
  search_files    - {"tool": "search_files", "query": "pattern", "file_type": "py"}
  list_files      - {"tool": "list_files", "path": "dir", "pattern": "*.py"}
  glob_files      - {"tool": "glob_files", "pattern": "**/*.py", "path": "src"}
  grep            - {"tool": "grep", "pattern": "regex", "file_type": "py"}
  write_file      - {"tool": "write_file", "path": "file", "content": "content"}
  edit_file       - {"tool": "edit_file", "path": "file", "old_string": "find", "new_string": "replace"}
  get_project_structure - {"tool": "get_project_structure"}

GIT:
  git_status      - {"tool": "git_status"}
  git_diff        - {"tool": "git_diff", "staged": false}
  git_log         - {"tool": "git_log", "count": 10}
  git_add         - {"tool": "git_add", "files": "file1.py file2.py"}
  git_commit      - {"tool": "git_commit", "message": "commit message"}
  git_branch      - {"tool": "git_branch", "name": "branch-name", "create": true}
  git_stash       - {"tool": "git_stash", "action": "push"}

WEB & INFO:
  web_search      - {"tool": "web_search", "query": "search query"}
  web_fetch       - {"tool": "web_fetch", "url": "https://example.com"}
  get_current_news - {"tool": "get_current_news", "topic": "topic"}
  get_gold_price  - {"tool": "get_gold_price", "currency": "USD"}
  get_weather     - {"tool": "get_weather", "city": "London"}
  get_current_time - {"tool": "get_current_time", "timezone": "UTC"}

UTILITIES:
  calculate       - {"tool": "calculate", "expression": "2 + 2"}
  run_command     - {"tool": "run_command", "command": "npm test"}
  github_search   - {"tool": "github_search", "query": "term", "search_type": "repos"}

MEMORY:
  save_memory     - {"tool": "save_memory", "content": "info", "category": "general"}
  recall_memory   - {"tool": "recall_memory", "query": "search term"}

TASKS:
  task_create     - {"tool": "task_create", "subject": "Title", "description": "details"}
  task_update     - {"tool": "task_update", "task_id": "1", "status": "completed"}
  task_list       - {"tool": "task_list"}
  task_get        - {"tool": "task_get", "task_id": "1"}

RULES:
1. For current events/news/prices: USE web_search
2. For code questions: read_file FIRST, never guess
3. For writing files: USE write_file or edit_file
4. Output ONLY valid JSON when calling a tool - no explanation before/after
'''

    def _parse_tool_call_from_text(self, text: str) -> tuple:
        """Try to extract a tool call from model's text output."""
        if not text:
            return None, None

        # Try to find JSON object with balanced braces
        # Start from first { and find matching }
        start = text.find('{')
        if start == -1:
            return None, None

        # Count braces to find the matching closing brace
        depth = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if depth != 0:
            return None, None

        json_str = text[start:end]

        try:
            data = json.loads(json_str)
            tool_name = data.get("tool") or data.get("name")
            if tool_name:
                # Get arguments - could be in "arguments" or directly in data
                args = data.get("arguments", {})
                # Handle case where arguments is a JSON string
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                if not args:
                    args = {k: v for k, v in data.items() if k not in ["tool", "name", "arguments"]}
                return tool_name, args
        except json.JSONDecodeError:
            pass

        return None, None

    def _format_tool_display(self, tool_name: str, args: dict, result: str = None) -> str:
        """Format tool call for display like Claude Code."""
        if tool_name == "read_file":
            path = args.get("path", "file")
            if result and not result.startswith("Error"):
                lines = len(result.split('\n'))
                return f"Read {path} ({lines} lines)"
            return f"Read {path}"

        elif tool_name == "list_files":
            path = args.get("path", ".") or "."
            pattern = args.get("pattern", "*")
            if pattern and pattern != "*":
                return f"List {path} ({pattern})"
            return f"List {path}"

        elif tool_name == "search_files":
            query = args.get("query", "")
            file_type = args.get("file_type", "")
            if file_type:
                return f"Search '{query}' in *.{file_type}"
            return f"Search '{query}'"

        elif tool_name == "run_command":
            cmd = args.get("command", "")
            if len(cmd) > 50:
                cmd = cmd[:47] + "..."
            return f"Run `{cmd}`"

        elif tool_name == "write_file":
            path = args.get("path", "file")
            return f"Write {path}"

        elif tool_name == "edit_file":
            path = args.get("path", "file")
            return f"Edit {path}"

        elif tool_name == "get_project_structure":
            return "Get project structure"

        elif tool_name == "web_search":
            query = args.get("query", "")
            if len(query) > 40:
                query = query[:37] + "..."
            return f"Search web: '{query}'"

        elif tool_name == "get_current_news":
            topic = args.get("topic", "")
            if len(topic) > 40:
                topic = topic[:37] + "..."
            return f"Get news: '{topic}'"

        elif tool_name == "get_weather":
            city = args.get("city", "")
            return f"Weather: {city}"

        elif tool_name == "get_current_time":
            tz = args.get("timezone", "UTC")
            return f"Time: {tz}"

        elif tool_name == "calculate":
            expr = args.get("expression", "")
            if len(expr) > 30:
                expr = expr[:27] + "..."
            return f"Calculate: {expr}"

        elif tool_name == "save_memory":
            cat = args.get("category", "general")
            return f"Save memory: [{cat}]"

        elif tool_name == "recall_memory":
            query = args.get("query", "")
            if query:
                return f"Recall: '{query}'"
            return "Recall memory"

        elif tool_name == "github_search":
            query = args.get("query", "")
            stype = args.get("search_type", "repos")
            return f"GitHub: {stype} '{query}'"

        # Git operations
        elif tool_name == "git_status":
            return "Git status"

        elif tool_name == "git_diff":
            staged = "staged " if args.get("staged") else ""
            file = args.get("file", "")
            desc = f"Git diff {staged}{file}".strip()
            return desc if desc != "Git diff" else "Git diff"

        elif tool_name == "git_log":
            count = args.get("count", 10)
            return f"Git log ({count} commits)"

        elif tool_name == "git_commit":
            msg = args.get("message", "")[:40]
            return f"Git commit: {msg}{'...' if len(args.get('message', '')) > 40 else ''}"

        elif tool_name == "git_add":
            files = args.get("files", ".")
            return f"Git add: {files[:30]}"

        elif tool_name == "git_branch":
            name = args.get("name", "")
            if args.get("create"):
                return f"Git create branch: {name}"
            elif args.get("switch"):
                return f"Git switch: {name}"
            return "Git branches"

        elif tool_name == "git_stash":
            action = args.get("action", "push")
            return f"Git stash {action}"

        # Enhanced file operations
        elif tool_name == "glob_files":
            pattern = args.get("pattern", "*")
            return f"Glob: {pattern}"

        elif tool_name == "grep":
            pattern = args.get("pattern", "")[:25]
            return f"Grep: '{pattern}'"

        # Web
        elif tool_name == "web_fetch":
            url = args.get("url", "")
            if len(url) > 40:
                url = url[:37] + "..."
            return f"Fetch: {url}"

        elif tool_name == "get_gold_price":
            cur = args.get("currency") or "USD"
            return f"Gold price: {cur}"

        # Task management
        elif tool_name == "task_create":
            subject = args.get("subject", "")[:30]
            return f"Create task: {subject}"

        elif tool_name == "task_update":
            task_id = args.get("task_id", "")
            status = args.get("status", "")
            return f"Update task #{task_id}" + (f" → {status}" if status else "")

        elif tool_name == "task_list":
            return "List tasks"

        elif tool_name == "task_get":
            task_id = args.get("task_id", "")
            return f"Get task #{task_id}"

        return f"{tool_name}()"

    def _validate_tool_call(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Validate tool call before execution."""
        # Schema for required parameters and types
        tool_schemas = {
            "read_file": {"required": ["path"], "types": {"path": str}},
            "write_file": {"required": ["path", "content"], "types": {"path": str, "content": str}},
            "edit_file": {"required": ["path", "old_string", "new_string"], "types": {"path": str}},
            "search_files": {"required": ["query"], "types": {"query": str}},
            "list_files": {"required": [], "types": {"path": str, "pattern": str}},
            "glob_files": {"required": ["pattern"], "types": {"pattern": str}},
            "grep": {"required": ["pattern"], "types": {"pattern": str}},
            "web_search": {"required": ["query"], "types": {"query": str}},
            "web_fetch": {"required": ["url"], "types": {"url": str}},
            "get_weather": {"required": ["city"], "types": {"city": str}},
            "get_current_time": {"required": [], "types": {"timezone": str}},
            "get_current_news": {"required": ["topic"], "types": {"topic": str}},
            "get_gold_price": {"required": [], "types": {"currency": str}},
            "calculate": {"required": ["expression"], "types": {"expression": str}},
            "run_command": {"required": ["command"], "types": {"command": str}},
            "git_commit": {"required": ["message"], "types": {"message": str}},
            "git_add": {"required": ["files"], "types": {"files": str}},
            "save_memory": {"required": ["content"], "types": {"content": str}},
            "recall_memory": {"required": [], "types": {"query": str}},
            "task_create": {"required": ["subject"], "types": {"subject": str}},
            "task_update": {"required": ["task_id"], "types": {"task_id": str}},
            "task_get": {"required": ["task_id"], "types": {"task_id": str}},
            "github_search": {"required": ["query"], "types": {"query": str}},
        }

        schema = tool_schemas.get(tool_name)
        if not schema:
            return True, ""  # Unknown tool, let it try

        # Check required parameters
        for param in schema.get("required", []):
            if param not in args or args[param] is None:
                return False, f"Missing required parameter: {param}"

        # Check types (basic validation)
        for param, expected_type in schema.get("types", {}).items():
            if param in args and args[param] is not None:
                if not isinstance(args[param], expected_type):
                    # Try to coerce
                    try:
                        args[param] = expected_type(args[param])
                    except (ValueError, TypeError):
                        return False, f"Parameter {param} should be {expected_type.__name__}"

        return True, ""

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name."""
        # Validate tool call
        valid, error = self._validate_tool_call(tool_name, args or {})
        if not valid:
            print(f"[TOOL] Validation failed for {tool_name}: {error}")
            return f"Error: {error}"

        # Log tool call
        args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in (args or {}).items())
        print(f"[TOOL] Calling: {tool_name}({args_str})")

        tool_map = {
            # File operations
            "read_file": read_file,
            "list_files": list_files,
            "search_files": search_files,
            "write_file": write_file,
            "edit_file": edit_file,
            "get_project_structure": get_project_structure,
            "glob_files": glob_files,
            "grep": grep,
            # Git operations
            "git_status": git_status,
            "git_diff": git_diff,
            "git_log": git_log,
            "git_commit": git_commit,
            "git_add": git_add,
            "git_branch": git_branch,
            "git_stash": git_stash,
            # Shell
            "run_command": run_command,
            # Web
            "web_search": web_search,
            "web_fetch": web_fetch,
            "get_current_news": get_current_news,
            "get_gold_price": get_gold_price,
            # Weather
            "get_weather": get_weather,
            # Time
            "get_current_time": get_current_time,
            # Math
            "calculate": calculate,
            # Memory
            "save_memory": save_memory,
            "recall_memory": recall_memory,
            # Task management
            "task_create": task_create,
            "task_update": task_update,
            "task_list": task_list,
            "task_get": task_get,
            # GitHub
            "github_search": github_search,
        }

        # Add media tools (lazy loaded)
        if tool_name in ["generate_image", "generate_video", "generate_music", "analyze_image"]:
            generate_image, generate_video, generate_music, analyze_image = _get_media_tools()
            tool_map.update({
                "generate_image": generate_image,
                "generate_video": generate_video,
                "generate_music": generate_music,
                "analyze_image": analyze_image,
            })

        if tool_name not in tool_map:
            print(f"[TOOL] ERROR: Unknown tool '{tool_name}'")
            return f"Unknown tool: {tool_name}. Available: {list(tool_map.keys())}"

        try:
            import time
            start = time.time()
            result = tool_map[tool_name](**args)
            duration = time.time() - start

            # Handle media generation results (return dict instead of string)
            if tool_name in ["generate_image", "generate_video", "generate_music", "analyze_image"]:
                if isinstance(result, dict):
                    if result.get("success"):
                        # Print media result in terminal
                        if self.ui and hasattr(self.ui, "print_media"):
                            media_type = {
                                "generate_image": "image",
                                "generate_video": "video",
                                "generate_music": "music",
                                "analyze_image": "analysis",
                            }.get(tool_name, "file")
                            if tool_name == "analyze_image":
                                # For analysis, just return the text
                                return result.get("analysis", "Analysis complete.")
                            self.ui.print_media(
                                media_type,
                                result.get("path", ""),
                                result.get("filename", "")
                            )
                        # Return a nice formatted string
                        path = result.get("path", "")
                        filename = result.get("filename", "")
                        return f"Generated {tool_name.replace('generate_', '')}: {filename}\nSaved to: {path}"
                    else:
                        return f"Error: {result.get('error', 'Unknown error')}"
                return str(result)

            result_preview = (result[:100] + "...") if len(result) > 100 else result
            result_preview = result_preview.replace("\n", " ")
            print(f"[TOOL] Completed: {tool_name} in {duration:.2f}s → {result_preview}")
            return result
        except Exception as e:
            print(f"[TOOL] ERROR: {tool_name} failed: {e}")
            return f"Error: {e}"

    def _call_model_with_timeout(self, messages, system_prompt, tools, timeout=120):
        """Call model with timeout using threading."""
        import threading
        import time

        result = {"response": None, "error": None}

        def call():
            try:
                result["response"] = self.provider.chat_with_tools(
                    messages=messages,
                    system=system_prompt,
                    tools=tools
                )
            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=call)
        thread.daemon = True
        thread.start()

        # Wait with periodic checks for interrupt
        start = time.time()
        while thread.is_alive():
            if self.ui and self.ui.stop_requested:
                return None  # Interrupted
            if time.time() - start > timeout:
                return None  # Timeout
            thread.join(timeout=0.5)  # Check every 0.5s

        if result["error"]:
            raise result["error"]
        return result["response"]

    def run(self, user_message: str, system_prompt: str, history: List = None) -> str:
        """Run agentic loop with tool calling."""
        import time
        start_time = time.time()
        tool_count = 0

        self.last_streamed = False
        if self.ui and hasattr(self.ui, "begin_turn"):
            self.ui.begin_turn()

        # Check if model supports native tools
        use_native = self._supports_native_tools()
        auto_tool_done = False
        agent_cfg = self.config.get("agent", {})

        if not use_native:
            # Add tools description to system prompt for prompt-based approach
            system_prompt = system_prompt + "\n\n" + self._get_tools_prompt()

        msg_lower = user_message.lower()
        explicit_search = [
            "web search", "search web", "search the web", "google", "look up",
            "find online", "search for", "browse"
        ]
        if any(s in msg_lower for s in explicit_search):
            try:
                tool_start = time.time()
                result = web_search(user_message, max_results=5)
                tool_duration = time.time() - tool_start
                tool_success = not (result and (result.startswith("Error") or result.startswith("error")))
                if self.ui:
                    self.ui.print_tool(self._format_tool_display("web_search", {"query": user_message}, result), success=tool_success)
                    if hasattr(self.ui, "record_tool"):
                        self.ui.record_tool(
                            "web_search",
                            self._format_tool_display("web_search", {"query": user_message}, result),
                            tool_duration,
                            args={"query": user_message},
                            result=result,
                            success=tool_success,
                        )
                return result
            except Exception:
                return "Search failed: unexpected error."

        if "gold" in msg_lower and "price" in msg_lower:
            import os
            if not (os.getenv("GOLDAPI_KEY") or os.getenv("GOLD_API_KEY")):
                return "Error: GOLDAPI_KEY not configured. Please set it in .env to fetch live gold prices."

        messages = []
        if history:
            for msg in history:
                if hasattr(msg, 'role'):
                    messages.append({"role": msg.role, "content": msg.content})
                else:
                    messages.append(msg)

        messages.append({"role": "user", "content": user_message})

        iteration = 0
        final_response = ""

        while iteration < self.max_iterations:
            iteration += 1

            if self.ui and self.ui.stop_requested:
                return "[dim]Stopped[/dim]"

            try:
                # Auto-run tools for current info, time, weather, etc.
                if not auto_tool_done:
                    auto_tool = self._detect_auto_tool(user_message)
                    if auto_tool:
                        tool_name, args = auto_tool
                        tool_start = time.time()
                        result = self._execute_tool(tool_name, args)
                        tool_duration = time.time() - tool_start
                        tool_count += 1
                        auto_tool_done = True
                        # Check if tool failed
                        tool_success = not (result and (result.startswith("Error") or result.startswith("error")))
                        if self.ui:
                            self.ui.print_tool(self._format_tool_display(tool_name, args, result), success=tool_success)
                            if hasattr(self.ui, "record_tool"):
                                self.ui.record_tool(
                                    tool_name,
                                    self._format_tool_display(tool_name, args, result),
                                    tool_duration,
                                    args=args,
                                    result=result,
                                    success=tool_success
                                )
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Tool result:\n{result}\n\n"
                                "Instructions:\n"
                                "1. If the result contains search results with titles/URLs/descriptions, synthesize a helpful answer from them.\n"
                                "2. If the result starts with 'Error:' or 'Search failed:', acknowledge the error briefly.\n"
                                "3. If you see 'No results found', say so briefly and suggest the user try a different query.\n"
                                "4. Include 1-2 source URLs when available.\n"
                                "5. Answer the original question directly and concisely."
                            )
                        })
                        # If web search completely failed, return the error directly
                        if tool_name == "web_search" and result and (
                            result.startswith("Search failed")
                            or result.startswith("Error")
                        ):
                            return result

                # If auto-tool already ran, just get LLM to synthesize - no more tools needed
                if auto_tool_done:
                    try:
                        from jarvis.providers import Message
                        synth_messages = [Message(role=m["role"], content=m["content"]) for m in messages]
                        synth_system = system_prompt + "\n\nAnswer the user's question based on the tool result above. Do NOT call any tools."
                        reply = self.provider.chat(
                            messages=synth_messages,
                            system=synth_system,
                            stream=True
                        )

                        content_parts = []
                        if hasattr(reply, "__iter__") and not isinstance(reply, (str, dict)):
                            for chunk in reply:
                                if self.ui and self.ui.stop_requested:
                                    break
                                content_parts.append(chunk)
                                if self.ui and hasattr(self.ui, "stream_text"):
                                    self.ui.stream_text(chunk)
                            if self.ui and hasattr(self.ui, "stream_done"):
                                self.ui.stream_done()
                            self.last_streamed = True
                        else:
                            content_parts = [str(reply) if isinstance(reply, str) else ""]

                        final_response = self._clean_content("".join(content_parts))
                        break
                    except Exception as e:
                        print(f"[agent] Synthesis error: {e}")
                        pass

                # Optional fast path when tools are unlikely
                if agent_cfg.get("fast_no_tools", True) and not self._likely_needs_tools(user_message):
                    try:
                        from jarvis.providers import Message
                        fast_messages = [Message(role=m["role"], content=m["content"]) for m in messages]
                        reply = self.provider.chat(
                            messages=fast_messages,
                            system=system_prompt,
                            stream=True
                        )

                        # Stream output live
                        content_parts = []
                        if hasattr(reply, "__iter__") and not isinstance(reply, (str, dict)):
                            for chunk in reply:
                                # Check for stop (Ctrl+C or ESC)
                                if self.ui:
                                    if self.ui.stop_requested:
                                        break
                                    if hasattr(self.ui, "check_escape_pressed") and self.ui.check_escape_pressed():
                                        self.ui.stop_requested = True
                                        break
                                content_parts.append(chunk)
                                if self.ui and hasattr(self.ui, "stream_text"):
                                    self.ui.stream_text(chunk)
                            if self.ui and hasattr(self.ui, "stream_done"):
                                self.ui.stream_done()
                            content = "".join(content_parts)
                            self.last_streamed = True
                        else:
                            # Non-stream reply fallback
                            content = None
                            if isinstance(reply, str):
                                content = reply
                            elif hasattr(reply, "message"):
                                msg = getattr(reply, "message")
                                if isinstance(msg, dict):
                                    content = msg.get("content")
                                else:
                                    content = getattr(msg, "content", None)
                            elif isinstance(reply, dict):
                                msg = reply.get("message")
                                if isinstance(msg, dict):
                                    content = msg.get("content")
                                else:
                                    content = msg
                            if self.ui and hasattr(self.ui, "stream_text") and content:
                                self.ui.stream_text(content)
                                self.ui.stream_done()
                                self.last_streamed = True

                        final_response = self._clean_content(content if content is not None else "")
                        break
                    except Exception:
                        # Fall back to tool-capable path
                        pass

                tools = ALL_TOOLS if use_native else None

                # Call model with timeout (interruptible)
                if self.ui:
                    self.ui.console.print("[dim]  Thinking...[/dim]", end="\r")

                timeout = self._get_timeout()
                response = self._call_model_with_timeout(messages, system_prompt, tools, timeout=timeout)

                # Clear the "Thinking..." line
                if self.ui:
                    self.ui.console.print("             ", end="\r")

                if response is None:
                    return "[dim]Stopped[/dim]"

                if self.ui and self.ui.stop_requested:
                    return "[dim]Stopped[/dim]"

                # Parse response
                msg = response.message if hasattr(response, 'message') else response.get('message', {})
                content = msg.content if hasattr(msg, 'content') else msg.get('content', '') or ''
                tool_calls = msg.tool_calls if hasattr(msg, 'tool_calls') else msg.get('tool_calls')

                # For native tool calling
                if use_native and tool_calls:
                    messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                    for call in tool_calls:
                        if self.ui and self.ui.stop_requested:
                            return "[dim]Stopped[/dim]"

                        tool_call_id = None
                        if hasattr(call, 'function'):
                            tool_name = call.function.name
                            args = call.function.arguments or {}
                            tool_call_id = getattr(call, "id", None)
                        else:
                            func = call.get('function', {})
                            tool_name = func.get('name', '')
                            args = func.get('arguments', {})
                            tool_call_id = call.get("id")

                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}

                        tool_start = time.time()
                        result = self._execute_tool(tool_name, args)
                        tool_duration = time.time() - tool_start
                        tool_count += 1
                        # Check if tool failed
                        tool_success = not (result and (result.startswith("Error") or result.startswith("error")))
                        if self.ui:
                            self.ui.print_tool(self._format_tool_display(tool_name, args, result), success=tool_success)
                            if hasattr(self.ui, "record_tool"):
                                self.ui.record_tool(
                                    tool_name,
                                    self._format_tool_display(tool_name, args, result),
                                    tool_duration,
                                    tool_call_id=tool_call_id,
                                    args=args,
                                    result=result,
                                    success=tool_success
                                )
                        tool_msg = {"role": "tool", "content": result}
                        if tool_call_id:
                            tool_msg["tool_call_id"] = tool_call_id
                        messages.append(tool_msg)

                    # Stream the final response after tool execution
                    try:
                        from jarvis.providers import Message
                        followup_system = system_prompt + "\n\nAnswer the user now. Do not call tools."
                        stream_messages = [Message(role=m["role"], content=m["content"]) for m in messages]
                        stream = self.provider.chat(
                            messages=stream_messages,
                            system=followup_system,
                            stream=True
                        )
                        parts = []
                        for chunk in stream:
                            # Check for stop (Ctrl+C or ESC)
                            if self.ui:
                                if self.ui.stop_requested:
                                    break
                                if hasattr(self.ui, "check_escape_pressed") and self.ui.check_escape_pressed():
                                    self.ui.stop_requested = True
                                    break
                            parts.append(chunk)
                            if self.ui and hasattr(self.ui, "stream_text"):
                                self.ui.stream_text(chunk)
                        if self.ui and hasattr(self.ui, "stream_done"):
                            self.ui.stream_done()
                        self.last_streamed = True
                        final_response = self._clean_content("".join(parts))
                        if self.ui and hasattr(self.ui, "print_tool_section"):
                            self.ui.print_tool_section()
                        break
                    except Exception:
                        # Fall back to normal loop if streaming fails
                        pass

                # Check if content contains JSON tool call (fallback for models that output JSON as text)
                elif content and ('"name"' in content or '"tool"' in content):
                    tool_name, args = self._parse_tool_call_from_text(content)

                    if tool_name:
                        result = self._execute_tool(tool_name, args)
                        tool_count += 1
                        if self.ui:
                            self.ui.print_tool(self._format_tool_display(tool_name, args, result))

                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Tool result:\n{result}\n\n"
                                "Instructions:\n"
                                "1. If the result contains search results with titles/URLs/descriptions, synthesize a helpful answer from them.\n"
                                "2. If the result starts with 'Error:' or 'Search failed:', acknowledge the error briefly.\n"
                                "3. If you see 'No results found', say so briefly and suggest the user try a different query.\n"
                                "4. Include 1-2 source URLs when available.\n"
                                "5. Answer the original question directly and concisely."
                            )
                        })
                    else:
                        # Couldn't parse JSON - remove JSON blob and return rest, or show error
                        # Find and remove the JSON object (handles nested braces)
                        clean = content
                        start = content.find('{')
                        if start != -1:
                            depth = 0
                            end = start
                            for i, char in enumerate(content[start:], start):
                                if char == '{':
                                    depth += 1
                                elif char == '}':
                                    depth -= 1
                                    if depth == 0:
                                        end = i + 1
                                        break
                            clean = (content[:start] + content[end:]).strip()

                        if clean:
                            final_response = clean
                        else:
                            final_response = "[dim]Model output malformed tool call. Try rephrasing your request.[/dim]"
                        break

                # No tool calls - final response
                elif content:
                    final_response = self._clean_content(content)
                    if self.ui and hasattr(self.ui, "print_tool_section"):
                        self.ui.print_tool_section()
                    break

                else:
                    break

            except Exception as e:
                return f"[red]Error: {e}[/red]"

        if iteration >= self.max_iterations:
            final_response += "\n[dim](max iterations)[/dim]"

        # Calculate elapsed time
        elapsed = time.time() - start_time

        # Format timing info
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.0f}s"

        # Add stats footer
        stats = f"\n[dim]({time_str}"
        if tool_count > 0:
            stats += f" · {tool_count} tool{'s' if tool_count > 1 else ''}"
        stats += ")[/dim]"

        response = final_response if final_response else "[dim]No response[/dim]"
        return response + stats

    def _clean_content(self, text: str) -> str:
        """Remove thinking tags and clean response."""
        if not text:
            return ""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()
