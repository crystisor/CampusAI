import logging
from typing import List, Dict, Any, Tuple, Optional
from src.config import config

logger = logging.getLogger(__name__)

class RerankerCandidate:
    """Represents a passage candidate from RAG or Web Search."""
    def __init__(self, text: str, source: str, metadata: Optional[Dict[str, Any]] = None, score: float = 0.0):
        self.text = text
        self.source = source  # e.g., "course_material" or "web_search"
        self.metadata = metadata or {}
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
            "score": self.score
        }

class Reranker:
    """
    Reranks candidate contexts using bge-reranker-v2-m3.
    Extracts the top 4-5 most relevant passages for the LLM.
    """

    def __init__(self, model_name: Optional[str] = None, top_k: Optional[int] = None, device: Optional[str] = None):
        self.model_name = model_name or config.reranker.model_name
        self.top_k = top_k or config.reranker.top_k
        self.device = device or config.reranker.device
        self._model = None
        self._tokenizer = None
        self._initialized = False

    def _lazy_init(self):
        """Load model on first call to save memory during bot startup."""
        if self._initialized:
            return
        
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            target_device = "cuda" if torch.cuda.is_available() and self.device in ["auto", "cuda"] else "cpu"
            logger.info(f"Loading reranker model '{self.model_name}' on {target_device}...")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            if target_device == "cuda":
                self._model = self._model.half().to("cuda")
            self._model.eval()
            self._target_device = target_device
            self._initialized = True
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load HuggingFace model '{self.model_name}' ({e}). Falling back to length & overlap ranker.")
            self._initialized = True
            self._model = None

    def rerank(
        self,
        query: str,
        candidates: List[RerankerCandidate],
        top_k: Optional[int] = None,
    ) -> List[RerankerCandidate]:
        """
        Reranks candidates and returns the top 4 to 5 highest scoring items.
        """
        k = top_k or self.top_k
        if not candidates:
            return []

        # If candidates are already <= k, return them directly
        if len(candidates) <= k:
            return candidates

        self._lazy_init()

        if self._model is not None and self._tokenizer is not None:
            try:
                import torch
                pairs = [[query, c.text] for c in candidates]
                with torch.no_grad():
                    inputs = self._tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self._target_device)
                    
                    scores = self._model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()

                for candidate, score in zip(candidates, scores):
                    candidate.score = float(score)

                sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
                return sorted_candidates[:k]
            except Exception as e:
                logger.error(f"Error during neural reranking: {e}. Using heuristic fallback.")

        # Fallback ranking (term overlap / existing score)
        return self._heuristic_rerank(query, candidates, k)

    def _heuristic_rerank(
        self,
        query: str,
        candidates: List[RerankerCandidate],
        top_k: int
    ) -> List[RerankerCandidate]:
        """Simple lexical and overlap fallback when neural reranker is offline."""
        q_words = set(query.lower().split())
        for c in candidates:
            c_words = set(c.text.lower().split())
            overlap = len(q_words.intersection(c_words))
            # Combine existing vector score with word overlap
            c.score = float(c.score * 0.5 + overlap * 0.5)

        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        return sorted_candidates[:top_k]
