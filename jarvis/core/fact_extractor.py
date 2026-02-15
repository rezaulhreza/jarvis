"""
Fact Extractor - Extracts key facts from conversations.

Analyzes chat history and extracts personal facts, preferences,
and context about the user to improve future responses.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import re


class FactExtractor:
    """Extracts and manages facts from conversations."""

    def __init__(self, facts_path: str = None):
        if facts_path:
            self.facts_path = Path(facts_path)
        else:
            from jarvis import get_data_dir
            self.facts_path = get_data_dir() / "memory" / "facts.md"

        self.facts_path.parent.mkdir(parents=True, exist_ok=True)

    def extract_facts(self, messages: list[dict], provider) -> list[str]:
        """
        Extract facts from a conversation using the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content',
                      or Message objects with .role and .content attributes
            provider: LLM provider to use for extraction

        Returns:
            List of extracted facts
        """
        if not messages or len(messages) < 2:
            return []

        # Safely extract role and content from mixed message types
        # (dict, Message objects, or other formats from tool-calling conversations)
        def _get_role(m):
            if isinstance(m, dict):
                return m.get("role", "unknown")
            return getattr(m, "role", "unknown")

        def _get_content(m):
            if isinstance(m, dict):
                c = m.get("content", "")
            else:
                c = getattr(m, "content", "")
            if not isinstance(c, str):
                c = str(c) if c else ""
            return c

        # Filter to user/assistant messages only (skip tool messages)
        filtered = [
            m for m in messages[-10:]
            if _get_role(m) in ("user", "assistant")
            and _get_content(m).strip()
        ]

        if len(filtered) < 2:
            return []

        # Build conversation text
        conversation = "\n".join([
            f"{_get_role(m).upper()}: {_get_content(m)[:500]}"
            for m in filtered
        ])

        # Prompt for fact extraction
        prompt = f"""Analyze this conversation and extract any NEW facts about the user.

Focus on:
- Personal info (name, location, job, company)
- Technical skills and preferences
- Projects they're working on
- Preferences and habits
- Goals and interests

CONVERSATION:
{conversation}

RULES:
- Only extract facts explicitly stated or strongly implied
- Be concise - one fact per line
- Format: "- Category: Fact"
- Examples: "- Job: Works at Kato as Senior Engineer"
- Return ONLY the facts, no explanations
- If no new facts found, return "NONE"

FACTS:"""

        try:
            response = provider.chat(
                messages=[{"role": "user", "content": prompt}],
                system="You extract facts from conversations. Be precise and concise.",
                stream=False,
                options={"num_predict": 300}
            )

            # Collect response
            result = ""
            for chunk in response:
                result += chunk

            result = result.strip()

            if result == "NONE" or not result:
                return []

            # Parse facts
            facts = []
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    facts.append(line[2:])
                elif line and ":" in line:
                    facts.append(line)

            return facts

        except Exception as e:
            print(f"[FactExtractor] Error: {e}")
            return []

    def load_existing_facts(self) -> str:
        """Load existing facts from file."""
        if self.facts_path.exists():
            return self.facts_path.read_text()
        return ""

    def merge_facts(self, new_facts: list[str], existing_content: str) -> str:
        """
        Merge new facts with existing facts, avoiding duplicates.

        Args:
            new_facts: List of new facts to add
            existing_content: Existing facts.md content

        Returns:
            Updated facts content
        """
        if not new_facts:
            return existing_content

        # Normalize existing facts for comparison
        existing_lower = existing_content.lower()

        # Filter out duplicates
        unique_facts = []
        for fact in new_facts:
            # Check if fact (or similar) already exists
            fact_key = fact.split(":")[0].lower().strip() if ":" in fact else fact.lower()[:30]
            fact_value = fact.split(":", 1)[1].lower().strip() if ":" in fact else fact.lower()

            # Skip if the key concept already exists
            if fact_value in existing_lower or fact_key in existing_lower:
                continue

            unique_facts.append(fact)

        if not unique_facts:
            return existing_content

        # Add new facts to the "Learned" section
        timestamp = datetime.now().strftime("%Y-%m-%d")

        if "## Learned" in existing_content:
            # Append to existing Learned section
            lines = existing_content.split("\n")
            new_lines = []
            found_learned = False

            for line in lines:
                new_lines.append(line)
                if line.strip() == "## Learned" and not found_learned:
                    found_learned = True
                    for fact in unique_facts:
                        new_lines.append(f"- {fact} ({timestamp})")

            return "\n".join(new_lines)
        else:
            # Create new Learned section at the end
            new_section = f"\n\n## Learned\n"
            for fact in unique_facts:
                new_section += f"- {fact} ({timestamp})\n"

            return existing_content.rstrip() + new_section

    def save_facts(self, content: str):
        """Save facts to file."""
        self.facts_path.write_text(content)

    def process_conversation(self, messages: list[dict], provider) -> int:
        """
        Process a conversation and extract/save facts.

        Args:
            messages: Conversation messages
            provider: LLM provider

        Returns:
            Number of new facts added
        """
        # Extract facts
        new_facts = self.extract_facts(messages, provider)

        if not new_facts:
            return 0

        # Load and merge
        existing = self.load_existing_facts()
        updated = self.merge_facts(new_facts, existing)

        # Save if changed
        if updated != existing:
            self.save_facts(updated)
            print(f"[FactExtractor] Added {len(new_facts)} new facts")
            return len(new_facts)

        return 0


# Singleton instance
_extractor: Optional[FactExtractor] = None


def get_fact_extractor() -> FactExtractor:
    """Get or create the fact extractor singleton."""
    global _extractor
    if _extractor is None:
        _extractor = FactExtractor()
    return _extractor
