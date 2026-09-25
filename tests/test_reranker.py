from src.core.reranker import Reranker, RerankerCandidate


def test_small_candidate_set_is_scored_and_filtered():
    reranker = Reranker()
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
