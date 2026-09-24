import re
import logging
from enum import Enum
from typing import Optional, Tuple
from src.core.ollama_client import OllamaClient
from src.config import config

logger = logging.getLogger(__name__)

class RoutingDecision(str, Enum):
    DIRECT = "DIRECT"    # Answer directly with model internal knowledge (casual, coding, math intuition)
    RAG = "RAG"          # Look up in subject course materials in Qdrant (lectures, assignments, slides)
    WEB = "WEB"          # Search the live web (current events, external docs, industry facts)
    HYBRID = "HYBRID"    # Compare course notes with external web information (syllabi vs real world)

ROUTER_SYSTEM_PROMPT = """You are an expert query classification router for a college study assistant.
Your job is to analyze the user's prompt and classify it into exactly ONE of the four categories:
1. DIRECT: The query is a general concept, coding question, mathematical derivation, greeting, or logical explanation that does NOT require college lecture notes or real-time internet search.
2. RAG: The query asks specifically about course notes, lecture slides, professor announcements, theorems from class, definitions, or homework problems from the course materials.
3. WEB: The query asks about real-world current events, external tools/libraries, modern software frameworks, or latest internet information not covered in academic slides.
4. HYBRID: The query explicitly or implicitly asks to COMPARE or COMBINE course materials with modern web/industry standards or external documentation.

Output ONLY the category name: DIRECT, RAG, WEB, or HYBRID. Do not explain.
"""

class IntentRouter:
    """Classifies user queries into DIRECT, RAG, WEB, or HYBRID using Arch-Router-1.5B."""

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client or OllamaClient()
        self.model_name = config.ollama.router_model

    async def route(self, user_prompt: str, subject_name: Optional[str] = None) -> Tuple[RoutingDecision, str]:
        """
        Classifies the prompt.
        Returns: (RoutingDecision, reason/raw_output)
        """
        context_hint = f"Subject Channel: {subject_name}\n" if subject_name else ""
        formatted_prompt = (
            f"{context_hint}User Prompt: \"{user_prompt}\"\n\nClassification:"
        )

        try:
            raw_response = await self.client.generate(
                prompt=formatted_prompt,
                model=self.model_name,
                system=ROUTER_SYSTEM_PROMPT,
                options={"temperature": 0.0, "max_tokens": 10},
            )
            cleaned = raw_response.strip().upper()
            
            # Match decision keywords
            for decision in [RoutingDecision.HYBRID, RoutingDecision.RAG, RoutingDecision.WEB, RoutingDecision.DIRECT]:
                if decision.value in cleaned:
                    return decision, cleaned

            # Fallback heuristic if model returned conversational output
            return self._heuristic_fallback(user_prompt), f"Heuristic parsed from '{cleaned}'"

        except Exception as e:
            logger.warning(f"Router model failed or not available ({e}), falling back to heuristic routing.")
            return self._heuristic_fallback(user_prompt), f"Fallback due to error: {e}"

    def _heuristic_fallback(self, prompt: str) -> RoutingDecision:
        """Rule-based heuristic when model is unavailable."""
        p_lower = prompt.lower()
        
        # Check hybrid keywords
        has_compare = any(w in p_lower for w in ["compare", "versus", "vs", "difference between course", "industry standard"])
        has_course = any(w in p_lower for w in ["slide", "lecture", "professor", "course", "exam", "syllabus", "notes", "chapter", "assignment"])
        has_web = any(w in p_lower for w in ["online", "web", "internet", "latest", "current", "news", "documentation", "github"])

        if (has_compare and has_course) or (has_course and has_web):
            return RoutingDecision.HYBRID
        if has_course:
            return RoutingDecision.RAG
        if has_web or any(w in p_lower for w in ["today", "latest version", "2024", "2025", "2026", "release"]):
            return RoutingDecision.WEB
        
        # Default to direct response
        return RoutingDecision.DIRECT
