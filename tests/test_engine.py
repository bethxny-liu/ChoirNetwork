"""Ranking contracts, using deterministic model outputs without downloads."""

import numpy as np
import pytest

from choirnetwork.engine import HymnIndex, HymnSimilarityEngine, load_engine


def sample_index():
    return HymnIndex(
        model_name="unused", numbers=list(range(1, 7)), variants=[""] * 6,
        slugs=[str(i) for i in range(1, 7)], titles=[f"Hymn {i}" for i in range(1, 7)],
        chunk_embeddings=np.array([[1 - i / 10, 0] for i in range(6)]),
        chunk_hymn_indices=np.arange(6), chunk_weights=np.full(6, 2.5),
        chunk_snippets=[f"Hymn {i}" for i in range(1, 7)], lyrics_preprocessed=[""] * 6,
    )


def engine_with_scores(scores, *, rerank=True):
    engine = HymnSimilarityEngine(sample_index(), use_reranker=rerank, use_lyric_boost=False)
    engine._encode_query = lambda query: np.array([1., 0.])

    class CrossEncoder:
        def predict(self, pairs):
            return np.array(scores)

    engine._get_cross_encoder = CrossEncoder
    return engine


@pytest.mark.parametrize("scores", [[-10, -9, -8, -7, 0, 1], [-4, -3, -2, -1, 0, 1]])
def test_requested_count_only_slices_the_ranking(scores):
    engine = engine_with_scores(scores)
    full = engine.search("grace", top_k=6)
    for k in (1, 2, 5):
        assert engine.search("grace", top_k=k) == full[:k]


def test_frozen_top_five_fallback_and_rerank_policy():
    fallback = engine_with_scores([-10, -9, -8, -7, 0, 1])
    accepted = engine_with_scores([-4, -3, -2, -1, 0, 1])
    assert [h.slug for h in fallback.search("grace", top_k=5)] == ["1", "2", "3", "4", "5"]
    assert [h.slug for h in accepted.search("grace", top_k=5)] == ["6", "5", "4", "3", "2"]


def test_low_reranker_scores_fill_from_dense_order_without_changing_top_five():
    engine = engine_with_scores([-6, -4, -3, -2, -1, 0])
    top_five = engine.search("grace", top_k=5)
    more = engine.search("grace", top_k=25)
    assert [h.slug for h in top_five] == ["6", "5", "4", "3", "2"]
    assert [h.slug for h in more[:5]] == [h.slug for h in top_five]
    assert [h.slug for h in more] == ["6", "5", "4", "3", "2", "1"]


def test_dense_scores_are_normalized_once():
    engine = engine_with_scores([], rerank=False)
    assert engine.search("grace", top_k=2)[0].score == pytest.approx(1.0)
    assert engine.search("grace", top_k=2)[1].score == pytest.approx(0.9)


def test_search_includes_best_lyric_stanza_without_changing_order():
    index = sample_index()
    index.chunk_embeddings = np.vstack([index.chunk_embeddings, [0.4, 0]])
    index.chunk_hymn_indices = np.append(index.chunk_hymn_indices, 0)
    index.chunk_weights = np.append(index.chunk_weights, 1.0)
    index.chunk_snippets.append("Amazing grace, how sweet the sound")
    engine = HymnSimilarityEngine(index, use_reranker=False, use_lyric_boost=False)
    engine._encode_query = lambda query: np.array([1., 0.])

    results = engine.search("grace", top_k=2)

    assert [h.slug for h in results] == ["1", "2"]
    assert results[0].snippet == "Amazing grace, how sweet the sound"
    assert results[1].snippet == ""


def test_experimental_query_overrides_are_separate():
    engine = engine_with_scores([0] * 6)
    seen = {}

    def encode(query):
        seen["retrieval"] = query
        return np.array([1., 0.])

    def rerank(query, candidates, scores):
        seen["rerank"] = query
        return []

    engine._encode_query = encode
    engine._rerank_candidates = rerank
    engine.search("title", retrieval_query="title and context")
    assert seen == {"retrieval": "title and context", "rerank": "title"}


@pytest.mark.parametrize("query,k", [(" ", 5), ("grace", 0), ("grace", 51)])
def test_invalid_search_inputs(query, k):
    with pytest.raises(ValueError):
        engine_with_scores([]).search(query, top_k=k)


def test_index_round_trip_and_missing_lyrics(tmp_path):
    index = sample_index()
    index.save(tmp_path)
    loaded = load_engine(tmp_path).index
    assert loaded.slugs == index.slugs
    np.testing.assert_array_equal(loaded.chunk_embeddings, index.chunk_embeddings)
    index.lyrics_preprocessed = None
    index.save(tmp_path)
    with pytest.raises(ValueError, match="missing lyrics"):
        load_engine(tmp_path)


def test_reject_legacy_index(tmp_path):
    (tmp_path / "metadata.json").write_text('{"index_version": 1}')
    with pytest.raises(ValueError, match="Unsupported index"):
        load_engine(tmp_path)


def test_build_save_load_and_search_without_model_downloads(monkeypatch, tmp_path):
    from choirnetwork.scraper import HymnRecord

    class Encoder:
        def __init__(self, name):
            pass

        def encode(self, texts, **kwargs):
            return np.array([[1., 0.] if "grace" in text else [0., 1.] for text in texts])

    monkeypatch.setattr("choirnetwork.engine.SentenceTransformer", Encoder)
    hymns = [
        HymnRecord(1, "", "Grace", "Grace and mercy", "https://example.com/1", "1"),
        HymnRecord(2, "", "Hope", "Hope and comfort", "https://example.com/2", "2"),
    ]
    HymnSimilarityEngine.from_raw_hymns(hymns).index.save(tmp_path)
    results = load_engine(tmp_path, use_reranker=False).search("grace", top_k=1)
    assert results[0].slug == "1"
