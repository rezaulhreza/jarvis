"""
LLM-based intent classification - replaces all hardcoded keywords.

This module provides intelligent intent classification using a fast LLM call,
replacing brittle keyword-based routing with adaptive, personalizable intent detection.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class Intent(Enum):
    """Supported intent types for routing."""
    CHAT = "chat"                    # Simple conversation, greetings, general Q&A
    SEARCH = "search"                # Web search needed for current info
    WEATHER = "weather"              # Weather query
    TIME_DATE = "time_date"          # Time/date info
    FILE_OP = "file_op"              # File operations (read, write, edit)
    CODE = "code"                    # Code generation/analysis
    GIT = "git"                      # Git operations
    SHELL = "shell"                  # Shell command execution
    CALCULATE = "calculate"          # Math/calculations
    RECALL = "recall"                # Memory/knowledge recall from RAG
    CONTROL = "control"              # System/device control
    VISION = "vision"                # Image analysis (what's in this image?)
    IMAGE_GEN = "image_gen"          # Image generation (draw, create, generate image)
    VIDEO_GEN = "video_gen"          # Video generation (make video, animate)
    MUSIC_GEN = "music_gen"          # Music generation (create music, song)
    NEWS = "news"                    # Current news/events
    FINANCE = "finance"              # Financial data (prices, stocks, crypto)
    UNKNOWN = "unknown"


class ReasoningLevel(Enum):
    """Reasoning depth levels for model selection."""
    FAST = "fast"           # Quick responses, simple queries
    BALANCED = "balanced"   # Default, moderate complexity
    DEEP = "deep"           # Complex reasoning, analysis


@dataclass
class ClassifiedIntent:
    """Result of intent classification."""
    intent: Intent
    confidence: float
    reasoning_level: ReasoningLevel
    requires_tools: bool
    suggested_tools: List[str] = field(default_factory=list)
    context_hint: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Ensure types are correct
        if isinstance(self.intent, str):
            try:
                self.intent = Intent(self.intent)
            except ValueError:
                self.intent = Intent.UNKNOWN
        if isinstance(self.reasoning_level, str):
            try:
                self.reasoning_level = ReasoningLevel(self.reasoning_level)
            except ValueError:
                self.reasoning_level = ReasoningLevel.BALANCED


class IntentClassifier:
    """
    Uses a fast LLM to classify user intent.

    This replaces hardcoded keyword matching with intelligent,
    context-aware intent detection.
    """

    # Compact classification prompt optimized for fast models
    CLASSIFICATION_PROMPT = '''Classify this user message. Output ONLY valid JSON.

Message: "{message}"

{context_section}

Output this exact JSON structure:
{{"intent": "<intent>", "confidence": <0.0-1.0>, "reasoning_level": "<level>", "requires_tools": <bool>, "suggested_tools": [<tools>]}}

Intents: chat, search, weather, time_date, file_op, code, git, shell, calculate, recall, news, finance, vision, image_gen, video_gen, music_gen, unknown

reasoning_level:
- "fast" = greetings, simple facts, quick info, yes/no questions
- "balanced" = most queries, explanations, moderate complexity
- "deep" = complex analysis, multi-step reasoning, detailed explanations

requires_tools = true when external data/actions needed (web, files, time, weather, etc.)

suggested_tools (only if requires_tools=true):
- search: web_search
- weather: get_weather
- time: get_current_time
- finance: get_gold_price, web_search
- news: get_current_news, web_search
- file_op: read_file, write_file, edit_file, list_files, search_files
- code: read_file, write_file, edit_file, search_files
- git: git_status, git_diff, git_log, git_commit, git_add
- shell: run_command
- calculate: calculate
- recall: recall_memory
- image_gen: generate_image
- video_gen: generate_video
- music_gen: generate_music
- vision: analyze_image

Examples:
"hello" -> {{"intent": "chat", "confidence": 0.99, "reasoning_level": "fast", "requires_tools": false, "suggested_tools": []}}
"what's the weather in London" -> {{"intent": "weather", "confidence": 0.95, "reasoning_level": "fast", "requires_tools": true, "suggested_tools": ["get_weather"]}}
"gold price" -> {{"intent": "finance", "confidence": 0.95, "reasoning_level": "fast", "requires_tools": true, "suggested_tools": ["get_gold_price"]}}
"read main.py" -> {{"intent": "file_op", "confidence": 0.95, "reasoning_level": "fast", "requires_tools": true, "suggested_tools": ["read_file"]}}
"explain quantum computing in detail" -> {{"intent": "chat", "confidence": 0.9, "reasoning_level": "deep", "requires_tools": false, "suggested_tools": []}}
"who is the president of France" -> {{"intent": "search", "confidence": 0.9, "reasoning_level": "fast", "requires_tools": true, "suggested_tools": ["web_search"]}}
"draw a cat in space" -> {{"intent": "image_gen", "confidence": 0.95, "reasoning_level": "balanced", "requires_tools": true, "suggested_tools": ["generate_image"]}}
"generate an image of sunset" -> {{"intent": "image_gen", "confidence": 0.95, "reasoning_level": "balanced", "requires_tools": true, "suggested_tools": ["generate_image"]}}
"what's in this image" -> {{"intent": "vision", "confidence": 0.95, "reasoning_level": "balanced", "requires_tools": true, "suggested_tools": ["analyze_image"]}}
"make a video of a running horse" -> {{"intent": "video_gen", "confidence": 0.95, "reasoning_level": "balanced", "requires_tools": true, "suggested_tools": ["generate_video"]}}
"create music for a podcast intro" -> {{"intent": "music_gen", "confidence": 0.95, "reasoning_level": "balanced", "requires_tools": true, "suggested_tools": ["generate_music"]}}

Output JSON only, no explanation:'''

    # Fallback patterns for when LLM classification fails
    FALLBACK_PATTERNS = {
        Intent.WEATHER: [r'\bweather\b', r'\btemperature\b', r'\bforecast\b'],
        Intent.TIME_DATE: [r'what\s*(?:\'s|is)?\s*(?:the\s+)?time\b', r'\bcurrent\s+time\b',
                          r'\btime\s+(?:now|in|for)\b', r'\bwhat\s+date\b', r'\bclock\b', r'\btimezone\b'],
        Intent.CALCULATE: [r'^[\d\.\+\-\*\/\(\)\s]+$', r'\bcalculate\b', r'\bcompute\b'],
        Intent.FILE_OP: [r'\bread\s+file\b', r'\bopen\s+file\b', r'\bedit\s+file\b', r'\bwrite\s+file\b',
                         r'\.(py|js|ts|go|rs|java|cpp|c|rb|php|yaml|yml|json|md|txt)\b'],
        Intent.GIT: [r'\bgit\s', r'\bcommit\b', r'\bpush\b', r'\bpull\b', r'\bbranch\b', r'\bmerge\b'],
        Intent.SHELL: [r'\brun\s+command\b', r'\bexecute\b', r'\bnpm\s', r'\bpip\s', r'\bpython\s'],
        Intent.SEARCH: [r'\bsearch\b', r'\bgoogle\b', r'\blook\s*up\b', r'\bfind\s+online\b',
                        r'\bbrowse\b', r'\bvisit\b', r'\bcheck\s+(out|this|the)\b',
                        r'\bgo\s+to\b', r'\bopen\s+(the\s+)?(url|link|site|website|page)\b',
                        r'https?://', r'\bwww\.', r'\bwebsite\b', r'\bwebpage\b',
                        # Bare domains with multi-part TLDs
                        r'\b[\w-]+\.(?:co\.uk|com\.au|org\.uk|co\.nz|co\.za|co\.in|com\.br)\b',
                        # Bare domains with single TLDs
                        r'\b[\w-]+\.(?:com|org|net|io|dev|app|ai|me|xyz|info|biz|edu)\b',
                        # "check <domain-like>"
                        r'\bcheck\s+\S+\.\w{2,}'],
        Intent.NEWS: [r'\bnews\b', r'\bheadline\b', r'\bbreaking\b'],
        Intent.FINANCE: [r'\bprice\b', r'\bstock\b', r'\bgold\b', r'\bsilver\b', r'\bbitcoin\b', r'\bbtc\b',
                         r'\bcrypto\b', r'\bmarket\b', r'\bforex\b'],
        Intent.CODE: [r'\bcode\b', r'\bfunction\b', r'\bclass\b', r'\brefactor\b', r'\bdebug\b'],
        # Multimodal intents
        Intent.IMAGE_GEN: [r'\bdraw\b', r'\bsketch\b', r'\bpaint\b', r'\bcreate\s+(an?\s+)?image\b',
                          r'\bgenerate\s+(an?\s+)?image\b', r'\bmake\s+(an?\s+)?(image|picture|art)\b',
                          r'\billustrat', r'\bdesign\s+(an?\s+)?(logo|poster|banner)\b'],
        Intent.VIDEO_GEN: [r'\bcreate\s+(a\s+)?video\b', r'\bgenerate\s+(a\s+)?video\b', r'\bmake\s+(a\s+)?video\b',
                          r'\banimate\b', r'\bvideo\s+of\b', r'\bturn\s+.+\s+into\s+(a\s+)?video\b',
                          r'\bclip\s+of\b', r'\banimation\s+of\b', r'\bshort\s+video\b',
                          r'\bi\s+want\s+(a\s+)?video\b', r'\bneed\s+(a\s+)?video\b'],
        Intent.MUSIC_GEN: [r'\bcreate\s+(a\s+)?music\b', r'\bgenerate\s+(a\s+)?music\b', r'\bmake\s+(a\s+)?song\b',
                          r'\bcompose\b', r'\bmusic\s+for\b', r'\bsoundtrack\b', r'\bjingle\b'],
        Intent.VISION: [r'\bwhat\'?s?\s+in\s+(this|the)\s+image\b', r'\banalyze\s+(this|the)\s+image\b',
                       r'\bdescribe\s+(this|the)\s+(image|picture|photo)\b', r'\bcan\s+you\s+see\b',
                       r'\blook\s+at\s+(this|the)\b', r'\bwhat\s+do\s+you\s+see\b'],
    }

    # Tools mapping for each intent
    INTENT_TOOLS = {
        Intent.WEATHER: ["get_weather"],
        Intent.TIME_DATE: ["get_current_time"],
        Intent.CALCULATE: ["calculate"],
        Intent.FILE_OP: ["read_file", "write_file", "edit_file", "list_files", "search_files"],
        Intent.GIT: ["git_status", "git_diff", "git_log", "git_commit", "git_add", "git_branch"],
        Intent.SHELL: ["run_command"],
        Intent.SEARCH: ["web_search", "web_fetch"],
        Intent.NEWS: ["get_current_news", "web_search"],
        Intent.FINANCE: ["get_gold_price", "web_search"],
        Intent.CODE: ["read_file", "write_file", "edit_file", "search_files", "get_project_structure"],
        Intent.RECALL: ["recall_memory"],
        # Multimodal
        Intent.IMAGE_GEN: ["generate_image"],
        Intent.VIDEO_GEN: ["generate_video"],
        Intent.MUSIC_GEN: ["generate_music"],
        Intent.VISION: ["analyze_image"],
    }

    # Current info signals that indicate need for fresh data
    CURRENT_INFO_SIGNALS = [
        r'\bcurrent\b', r'\blatest\b', r'\btoday\b', r'\bright\s+now\b', r'\brecent\b',
        r'\b202[4-9]\b', r'\b203\d\b',  # Years 2024-2039
        r'\bpresident\b', r'\bprime\s+minister\b', r'\bceo\b', r'\bchairman\b',
        r'\bscore\b', r'\bmatch\b', r'\bgame\b', r'\btournament\b',
        r'\breleased?\b', r'\blaunched?\b', r'\bannounced?\b'
    ]

    def __init__(self, provider=None, config: dict = None):
        """
        Initialize the intent classifier.

        Args:
            provider: LLM provider for classification (uses fast model)
            config: Configuration dict with intent settings
        """
        self.provider = provider
        self.config = config or {}
        self._cache: Dict[str, tuple] = {}  # key -> (timestamp, result)
        self._cache_ttl = self.config.get("intent", {}).get("cache_ttl", 300)
        self._cache_max = 500  # Max entries before eviction

    def _get_context_section(self, context: List[dict] = None) -> str:
        """Build context section for the prompt if context is provided."""
        if not context:
            return ""

        # Get last 2 messages for context
        recent = context[-2:] if len(context) > 2 else context
        if not recent:
            return ""

        context_lines = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:100]  # Truncate for speed
            context_lines.append(f"{role}: {content}")

        return f"Recent context:\n" + "\n".join(context_lines)

    async def classify(self, message: str, context: List[dict] = None, use_llm: bool = False) -> ClassifiedIntent:
        """
        Classify user message intent.

        Uses fast heuristic matching by default for instant results.
        LLM classification is optional and only used when explicitly requested
        or when heuristics return low confidence.

        Args:
            message: User's message to classify
            context: Optional conversation context
            use_llm: Force LLM classification (slower but more accurate)

        Returns:
            ClassifiedIntent with detected intent and metadata
        """
        if not message or not message.strip():
            return ClassifiedIntent(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                reasoning_level=ReasoningLevel.FAST,
                requires_tools=False
            )

        # Check cache first (with TTL)
        import time as _time
        cache_key = message.lower().strip()
        if cache_key in self._cache:
            ts, cached_result = self._cache[cache_key]
            if _time.time() - ts < self._cache_ttl:
                return cached_result
            else:
                del self._cache[cache_key]

        # Use fast heuristic classification by default (instant)
        result = self._classify_heuristic(message)

        # Only use LLM if explicitly requested AND provider available AND heuristic confidence is low
        intent_cfg = self.config.get("intent", {})
        llm_enabled = intent_cfg.get("llm_enabled", False)  # LLM disabled by default for speed

        if use_llm or (llm_enabled and result.confidence < 0.6):
            if self.provider and intent_cfg.get("enabled", True):
                try:
                    llm_result = await self._classify_with_llm(message, context)
                    if llm_result.confidence >= intent_cfg.get("confidence_threshold", 0.7):
                        result = llm_result
                except Exception as e:
                    print(f"[IntentClassifier] LLM classification failed: {e}")

        # Evict oldest entries if cache is full
        if len(self._cache) >= self._cache_max:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[cache_key] = (_time.time(), result)
        return result

    def classify_sync(self, message: str, context: List[dict] = None) -> ClassifiedIntent:
        """
        Synchronous version of classify for non-async contexts.

        Always uses fast heuristic classification for instant results.
        This is the preferred method for UI responsiveness.
        """
        # Check cache first (with TTL)
        import time as _time
        cache_key = message.lower().strip()
        if cache_key in self._cache:
            ts, cached_result = self._cache[cache_key]
            if _time.time() - ts < self._cache_ttl:
                return cached_result
            else:
                del self._cache[cache_key]

        # Use heuristic classification (instant, no LLM call)
        result = self._classify_heuristic(message)
        if len(self._cache) >= self._cache_max:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[cache_key] = (_time.time(), result)
        return result

    async def _classify_with_llm(self, message: str, context: List[dict] = None) -> ClassifiedIntent:
        """Use LLM to classify intent."""
        context_section = self._get_context_section(context)
        prompt = self.CLASSIFICATION_PROMPT.format(
            message=message[:500],  # Truncate for speed
            context_section=context_section
        )

        try:
            from jarvis.providers import Message

            # Use fast model for classification
            response = self.provider.chat(
                messages=[Message(role="user", content=prompt)],
                system="You are an intent classifier. Output only valid JSON.",
                stream=False
            )

            # Parse response
            if hasattr(response, '__iter__') and not isinstance(response, str):
                response = ''.join(response)

            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return ClassifiedIntent(
                    intent=Intent(data.get("intent", "unknown")),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning_level=ReasoningLevel(data.get("reasoning_level", "balanced")),
                    requires_tools=bool(data.get("requires_tools", False)),
                    suggested_tools=data.get("suggested_tools", []),
                    context_hint=data.get("context_hint"),
                    raw_response=data
                )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[IntentClassifier] Parse error: {e}")

        # Return low-confidence result on failure
        return ClassifiedIntent(
            intent=Intent.UNKNOWN,
            confidence=0.3,
            reasoning_level=ReasoningLevel.BALANCED,
            requires_tools=False
        )

    def _classify_heuristic(self, message: str) -> ClassifiedIntent:
        """
        Fallback heuristic classification using patterns.

        This provides a reliable fallback when LLM classification
        is unavailable or fails.
        """
        msg_lower = message.lower().strip()

        # Check each intent pattern
        for intent, patterns in self.FALLBACK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, msg_lower, re.IGNORECASE):
                    return ClassifiedIntent(
                        intent=intent,
                        confidence=0.8,
                        reasoning_level=self._determine_reasoning_level(message, intent),
                        requires_tools=intent in self.INTENT_TOOLS,
                        suggested_tools=self.INTENT_TOOLS.get(intent, [])
                    )

        # Check for current info needs
        needs_current_info = any(
            re.search(pattern, msg_lower, re.IGNORECASE)
            for pattern in self.CURRENT_INFO_SIGNALS
        )

        if needs_current_info:
            return ClassifiedIntent(
                intent=Intent.SEARCH,
                confidence=0.75,
                reasoning_level=ReasoningLevel.FAST,
                requires_tools=True,
                suggested_tools=["web_search"]
            )

        # Default to chat
        return ClassifiedIntent(
            intent=Intent.CHAT,
            confidence=0.6,
            reasoning_level=self._determine_reasoning_level(message, Intent.CHAT),
            requires_tools=False
        )

    def _determine_reasoning_level(self, message: str, intent: Intent) -> ReasoningLevel:
        """Determine appropriate reasoning level based on message and intent."""
        msg_lower = message.lower()

        # Check user overrides from config
        overrides = self.config.get("intent", {}).get("reasoning_overrides", {})
        for pattern, level in overrides.items():
            if re.search(pattern, msg_lower, re.IGNORECASE):
                try:
                    return ReasoningLevel(level)
                except ValueError:
                    pass

        # Fast indicators
        fast_indicators = [
            r'^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|bye)\b',
            r'\bquick\b', r'\bbrief\b', r'\bshort\b',
            r'^\w+\?$',  # Single word questions
        ]
        for pattern in fast_indicators:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                return ReasoningLevel.FAST

        # Deep indicators
        deep_indicators = [
            r'\bexplain\b.*\bin\s+detail\b', r'\bthink\b.*\bthrough\b',
            r'\banalyze\b', r'\bcompare\b.*\band\b.*\bcontrast\b',
            r'\bwhy\b.*\?.*\bwhy\b', r'\bstep\s+by\s+step\b',
            r'\bcomprehensive\b', r'\bthorough\b', r'\bdetailed\b',
        ]
        for pattern in deep_indicators:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                return ReasoningLevel.DEEP

        # Intent-based defaults
        if intent in [Intent.WEATHER, Intent.TIME_DATE, Intent.CALCULATE]:
            return ReasoningLevel.FAST
        if intent in [Intent.CODE, Intent.FILE_OP] and len(message) > 100:
            return ReasoningLevel.DEEP
        # Multimodal tasks use balanced (need API call, not super fast but not deep reasoning)
        if intent in [Intent.IMAGE_GEN, Intent.VIDEO_GEN, Intent.MUSIC_GEN, Intent.VISION]:
            return ReasoningLevel.BALANCED

        return ReasoningLevel.BALANCED

    def requires_tools(self, intent: ClassifiedIntent) -> bool:
        """Check if the classified intent requires tool execution."""
        return intent.requires_tools or intent.intent in self.INTENT_TOOLS

    def get_suggested_tools(self, intent: ClassifiedIntent) -> List[str]:
        """Get suggested tools for the classified intent."""
        if intent.suggested_tools:
            return intent.suggested_tools
        return self.INTENT_TOOLS.get(intent.intent, [])

    def clear_cache(self):
        """Clear the classification cache."""
        self._cache.clear()
