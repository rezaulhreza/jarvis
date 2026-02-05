"""
Multi-Model Analysis Skill

Run queries through multiple AI models simultaneously and combine their insights.
Uses Chutes AI to access different model types for comprehensive analysis.

Model Types:
- fast: Quick responses (gemma-3-4b)
- reasoning: Deep logical analysis (DeepSeek-V3)
- code: Code generation/analysis (Qwen2.5-Coder)
- thinking: Step-by-step reasoning (Qwen3-235B-Thinking)
- vision: Image understanding (Qwen2.5-VL)
"""

import asyncio
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import os


# Analysis profiles - which models to use for different analysis types
ANALYSIS_PROFILES = {
    "comprehensive": {
        "description": "Full analysis using all model types",
        "models": ["fast", "reasoning", "code", "thinking"],
    },
    "quick": {
        "description": "Fast analysis using lightweight models",
        "models": ["fast", "reasoning"],
    },
    "technical": {
        "description": "Technical/code focused analysis",
        "models": ["code", "reasoning", "thinking"],
    },
    "reasoning": {
        "description": "Deep reasoning and logic analysis",
        "models": ["reasoning", "thinking"],
    },
}


def get_chutes_provider():
    """Get configured Chutes provider."""
    try:
        from jarvis.providers import get_provider
        from jarvis.assistant import load_config
        config = load_config()
        return get_provider("chutes", config=config)
    except Exception as e:
        print(f"[Multi-Model] Error loading Chutes provider: {e}")
        return None


def analyze_with_model(
    provider,
    model_type: str,
    query: str,
    system_prompt: str = None,
) -> dict:
    """Run analysis with a specific model type."""
    from jarvis.providers.base import Message

    # Get model for task
    model_name = provider.get_model_for_task(model_type)

    # Save original model
    original_model = provider.model

    try:
        # Switch to specific model
        provider.model = model_name

        # Create system prompt based on model type
        if not system_prompt:
            system_prompts = {
                "fast": "You are a quick-response assistant. Provide concise, direct answers.",
                "reasoning": "You are an analytical assistant. Break down problems logically and consider multiple perspectives.",
                "code": "You are a code expert. Analyze technical aspects, suggest implementations, and identify potential issues.",
                "thinking": "You are a deep thinker. Show your step-by-step reasoning process clearly.",
            }
            system_prompt = system_prompts.get(model_type, "You are a helpful assistant.")

        # Run query
        messages = [Message(role="user", content=query)]

        # Use non-streaming for simpler result handling
        response = provider.chat(messages, system=system_prompt, stream=False)

        # Handle generator response (Python generators with yield + return)
        # If it's a generator, collect all chunks
        import types
        if isinstance(response, types.GeneratorType):
            response = "".join(list(response))

        return {
            "model_type": model_type,
            "model_name": model_name,
            "response": response,
            "success": True,
        }
    except Exception as e:
        return {
            "model_type": model_type,
            "model_name": model_name,
            "response": f"Error: {str(e)}",
            "success": False,
        }
    finally:
        # Restore original model
        provider.model = original_model


async def analyze_parallel(
    query: str,
    profile: str = "comprehensive",
    custom_models: list = None,
) -> dict:
    """
    Run multi-model analysis in parallel.

    Args:
        query: The question or topic to analyze
        profile: Analysis profile (comprehensive, quick, technical, reasoning)
        custom_models: Optional list of specific model types to use

    Returns:
        Dict with results from each model and a synthesis
    """
    provider = get_chutes_provider()
    if not provider:
        return {
            "error": "Chutes provider not configured. Set CHUTES_API_KEY.",
            "results": [],
        }

    # Get models to use
    if custom_models:
        model_types = custom_models
    else:
        profile_config = ANALYSIS_PROFILES.get(profile, ANALYSIS_PROFILES["comprehensive"])
        model_types = profile_config["models"]

    # Run analyses in parallel using ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=len(model_types)) as executor:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                executor,
                analyze_with_model,
                provider,
                model_type,
                query,
                None,
            )
            for model_type in model_types
        ]
        results = await asyncio.gather(*futures)

    # Build synthesis from successful results
    successful = [r for r in results if r["success"]]

    synthesis = None
    if len(successful) >= 2:
        # Create synthesis using the reasoning model
        synthesis_prompt = f"""Based on these different AI perspectives on the query "{query}":

"""
        for r in successful:
            synthesis_prompt += f"**{r['model_type'].upper()} Model ({r['model_name']}):**\n{r['response']}\n\n"

        synthesis_prompt += """
Synthesize these perspectives into a comprehensive answer that:
1. Identifies key points of agreement
2. Notes any contradictions or different viewpoints
3. Provides a balanced final conclusion

Keep the synthesis concise but complete."""

        try:
            synthesis_result = analyze_with_model(
                provider,
                "reasoning",
                synthesis_prompt,
                "You are an expert at synthesizing multiple AI perspectives into clear, actionable insights.",
            )
            if synthesis_result["success"]:
                synthesis = synthesis_result["response"]
        except Exception as e:
            print(f"[Multi-Model] Synthesis error: {e}")

    return {
        "query": query,
        "profile": profile,
        "models_used": model_types,
        "results": results,
        "synthesis": synthesis,
        "success_count": len(successful),
        "total_count": len(results),
    }


def multi_model_analyze(
    query: str,
    profile: str = "comprehensive",
) -> str:
    """
    Analyze a query using multiple AI models.

    This runs the query through different specialized models (fast, reasoning,
    code, thinking) and synthesizes their responses into comprehensive insights.

    Args:
        query: The question or topic to analyze
        profile: Analysis profile - "comprehensive" (all models), "quick" (fast),
                "technical" (code-focused), or "reasoning" (logic-focused)

    Returns:
        Combined analysis from multiple models with synthesis
    """
    # Run async function synchronously
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(analyze_parallel(query, profile))
        loop.close()
    except Exception as e:
        return f"Multi-model analysis failed: {str(e)}"

    if "error" in result:
        return result["error"]

    # Format output
    output = f"# Multi-Model Analysis\n\n"
    output += f"**Query:** {result['query']}\n"
    output += f"**Profile:** {result['profile']} ({result['success_count']}/{result['total_count']} models succeeded)\n\n"

    output += "---\n\n"

    for r in result["results"]:
        status = "✅" if r["success"] else "❌"
        output += f"## {status} {r['model_type'].upper()} Model\n"
        output += f"*Model: {r['model_name']}*\n\n"
        output += f"{r['response']}\n\n"
        output += "---\n\n"

    if result.get("synthesis"):
        output += "## 🔮 Synthesis\n\n"
        output += result["synthesis"]

    return output


def list_analysis_profiles() -> str:
    """List available multi-model analysis profiles."""
    output = "# Available Analysis Profiles\n\n"
    for name, config in ANALYSIS_PROFILES.items():
        models = ", ".join(config["models"])
        output += f"**{name}**: {config['description']}\n"
        output += f"  Models: {models}\n\n"
    return output


# Skill info for registration
SKILL_INFO = {
    "name": "multi_model_analyze",
    "function": multi_model_analyze,
    "description": "Analyze a query using multiple AI models for comprehensive insights",
    "parameters": {
        "query": "string - The question or topic to analyze",
        "profile": "string - Analysis profile: comprehensive, quick, technical, reasoning (default: comprehensive)"
    }
}
