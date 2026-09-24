import logging
from typing import List, Dict, Any, Optional, Tuple
from src.core.ollama_client import OllamaClient
from src.core.router import IntentRouter, RoutingDecision
from src.core.reranker import Reranker, RerankerCandidate
from src.core.search import WebSearchEngine
from src.rag.qdrant_manager import QdrantManager
from src.config import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are CampusAI, an elite academic and study assistant for college students.
Subject context: {subject_name}

Your goal is to answer the student's question with utmost mathematical precision, clear logical structuring, and verified citations.
When context is provided from course materials or the web, integrate it seamlessly and refer directly to theorem names, lecture slides, page numbers, or web sources.
Discord does not render LaTeX. Do NOT use LaTeX tags or commands (no $$, $, \frac, \sum, \sqrt).
Instead, write all formulas in clean, human-readable plain text using Unicode math symbols
(such as s², σ², μ, x, √, Σ, 1/N, etc.) or single-line code blocks (`s = √(s²)`).
If you are comparing course material with web knowledge, clearly state any differences or nuances between academic theory and real-world implementations.
"""

class RAGPipeline:
    """
    End-to-End Orchestrator:
    1. Intent Classification (Arch-Router)
    2. Retrieval (Qdrant bge-m3 / Web Search)
    3. Cross-Encoder Reranking (bge-reranker-v2-m3 -> Top 4-5)
    4. Synthesis (Spark-X2.5-4b-Q8_0)
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        router: Optional[IntentRouter] = None,
        reranker: Optional[Reranker] = None,
        qdrant_manager: Optional[QdrantManager] = None,
        search_engine: Optional[WebSearchEngine] = None,
    ):
        self.ollama = ollama_client or OllamaClient()
        self.router = router or IntentRouter(self.ollama)
        self.reranker = reranker or Reranker()
        self.qdrant = qdrant_manager or QdrantManager()
        self.search = search_engine or WebSearchEngine()

    async def process_query(
        self,
        query: str,
        subject_id: str,
        subject_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes full query pipeline and returns response with metadata.
        """
        sub_name = subject_name or subject_id
        
        # 1. Routing classification
        decision, route_reason = await self.router.route(query, subject_name=sub_name)
        logger.info(f"Routed query '{query[:40]}...' to {decision.value} ({route_reason})")

        candidates: List[RerankerCandidate] = []

        # 2. Retrieval according to decision
        if decision in [RoutingDecision.RAG, RoutingDecision.HYBRID]:
            # Retrieve from Qdrant
            try:
                query_vector = await self.ollama.get_embedding(query, model=config.ollama.embedding_model)
                if query_vector:
                    course_candidates = await self.qdrant.search(subject_id, query_vector, limit=8)
                    candidates.extend(course_candidates)
            except Exception as e:
                logger.error(f"RAG retrieval error: {e}")

        if decision in [RoutingDecision.WEB, RoutingDecision.HYBRID]:
            # Retrieve from Web Search
            try:
                web_candidates = await self.search.search(query, max_results=5)
                candidates.extend(web_candidates)
            except Exception as e:
                logger.error(f"Web search retrieval error: {e}")

        # 3. Rerank to top 4-5 items
        top_candidates: List[RerankerCandidate] = []
        if candidates:
            # Explicitly select top 4-5 as specified
            top_candidates = self.reranker.rerank(query, candidates, top_k=5)
            logger.info(f"Reranked {len(candidates)} candidates down to top {len(top_candidates)}")

        # 4. Construct context block
        context_str = ""
        if top_candidates:
            context_blocks = []
            for idx, c in enumerate(top_candidates, 1):
                context_blocks.append(f"--- Reference [{idx}] ({c.source}) ---\n{c.text}")
            context_str = "\n\n".join(context_blocks)

        # 5. Build prompt for Spark-X2.5-4b-Q8_0
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(subject_name=sub_name)
        
        if context_str:
            final_user_prompt = (
                f"You have been provided with the following verified reference context:\n\n"
                f"{context_str}\n\n"
                f"Based on the references above and your knowledge, answer the student's question:\n"
                f"Student Question: {query}"
            )
        else:
            final_user_prompt = query

        # 6. Generate answer with Spark-X2.5-4b-Q8_0
        response_text = await self.ollama.generate(
            prompt=final_user_prompt,
            model=config.ollama.llm_model,
            system=system_prompt,
            options={"temperature": 0.3}
        )

        return {
            "answer": response_text,
            "decision": decision.value,
            "route_reason": route_reason,
            "top_contexts": [c.to_dict() for c in top_candidates],
        }
