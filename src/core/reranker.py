import logging
import threading
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
        self.max_length = config.reranker.max_length
        self.score_threshold = config.reranker.score_threshold
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._lock = threading.Lock()

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
        # Worker calls may overlap, including after an awaiting task is cancelled.
        # Serialize model loading and inference on the worker, never the event loop.
        with self._lock:
            return self._rerank(query, candidates, top_k)

    def _rerank(
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

        self._lazy_init()

        if self._model is not None and self._tokenizer is not None:
            try:
                import torch
                pairs = [[query, c.text] for c in candidates]
                with torch.no_grad():
                    inputs = self._tokenizer(
                        [pair[0] for pair in pairs],
                        [pair[1] for pair in pairs],
                        padding=True,
                        truncation="only_second",
                        max_length=self.max_length,
                        stride=min(128, self.max_length // 4),
                        return_overflowing_tokens=True,
                        return_tensors="pt"
                    )
                    # Pages can exceed the token limit; score every overlapping
                    # passage instead of silently discarding formulas near the end.
                    owners = inputs.pop("overflow_to_sample_mapping").tolist()
                    scores = [float("-inf")] * len(candidates)
                    for start in range(0, len(owners), 8):
                        batch = {key: value[start:start + 8].to(self._target_device)
                                 for key, value in inputs.items()}
                        window_scores = self._model(**batch, return_dict=True).logits.view(-1).float().cpu().tolist()
                        for owner, score in zip(owners[start:start + 8], window_scores):
                            scores[owner] = max(scores[owner], score)

                for candidate, score in zip(candidates, scores):
                    candidate.score = float(score)

                sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
                selected = [c for c in sorted_candidates if c.score >= self.score_threshold][:k]
                logger.info(
                    "Reranker raw scores: min=%.3f max=%.3f threshold=%s kept=%d/%d",
                    min(scores), max(scores), self.score_threshold, len(selected), len(candidates),
                )
                return selected
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
        return [c for c in sorted_candidates if c.score >= self.score_threshold][:top_k]
