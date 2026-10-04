from src.core.reranker import Reranker, RerankerCandidate


def test_small_candidate_set_is_scored_and_filtered():
    reranker = Reranker()
    reranker.score_threshold = 0.0
    # Exercise the deterministic fallback without loading model weights.
    reranker._initialized = True
    candidates = [
        RerankerCandidate("unrelated", "course_material", score=-2.0),
        RerankerCandidate("calculus", "course_material", score=0.0),
        RerankerCandidate("other", "course_material", score=0.0),
    ]

    ranked = reranker.rerank("calculus", candidates)

    assert [candidate.text for candidate in ranked] == ["calculus", "other"]
    assert [candidate.score for candidate in ranked] == [0.5, 0.0]


def test_neural_ranking_uses_formula_beyond_first_window():
    import torch
    from unittest.mock import Mock

    reranker = Reranker()
    reranker._initialized = True
    reranker._target_device = "cpu"
    reranker.score_threshold = 0.0
    reranker._tokenizer = Mock(return_value={
        "input_ids": torch.ones((3, 8), dtype=torch.long),
        "attention_mask": torch.ones((3, 8), dtype=torch.long),
        "overflow_to_sample_mapping": torch.tensor([0, 0, 1]),
    })
    reranker._model = Mock(return_value=Mock(logits=torch.tensor([[-8.0], [4.0], [1.0]])))
    candidates = [RerankerCandidate("long formula page", "course_material"),
                  RerankerCandidate("other page", "course_material")]
    assert reranker.rerank("formula?", candidates, top_k=1) == [candidates[0]]
    assert candidates[0].score == 4.0
    assert reranker._tokenizer.call_args.kwargs["return_overflowing_tokens"] is True
