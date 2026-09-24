import numpy as np
import pytest

from choirnetwork.engine import HymnIndex
from choirnetwork.scraper import HymnRecord, save_hymns
from experiments.ranking import RankingExperiment, lyric_mask_for_corpus


def sample_index():
    return HymnIndex(
        model_name="unused", numbers=[1, 2], variants=["", ""], slugs=["1", "2"],
        titles=["Title", "No lyrics"], chunk_snippets=["Title", "First stanza", "Best stanza", "No lyrics"],
        chunk_hymn_indices=np.array([0, 0, 0, 1]), chunk_weights=np.ones(4),
    )


def test_natural_query_keeps_negation_and_punctuation():
    engine = RankingExperiment(sample_index(), natural_text=True)
    seen = []

    class Encoder:
        def encode(self, texts, **kwargs):
            seen.extend(texts)
            return np.ones((len(texts), 2))

    engine._model = Encoder()
    engine._encode_query("  Love is not\nJealous! ")
    assert seen == ["Love is not Jealous!"]


def test_stanza_selection_excludes_title_without_using_weights():
    index = sample_index()
    scores = np.array([0.99, 0.3, 0.6, 0.8])
    baseline = RankingExperiment(index)
    stanza = RankingExperiment(index, stanza_only=True, lyric_mask=np.array([False, True, True, False]))
    assert baseline._best_chunk_for_hymn(0, scores) == "Title"
    assert stanza._best_chunk_for_hymn(0, scores) == "Best stanza"
    assert stanza._best_chunk_for_hymn(1, scores) == "No lyrics"


def test_stanza_mask_requires_the_original_corpus(tmp_path):
    path = tmp_path / "hymns.json"
    save_hymns([
        HymnRecord(1, "", "Title", "First stanza\n\nBest stanza", "example", "1"),
        HymnRecord(2, "", "No lyrics", "", "example", "2"),
    ], path)
    index = sample_index()
    np.testing.assert_array_equal(lyric_mask_for_corpus(index, path), [False, True, True, False])
    index.chunk_snippets[1] = "Different corpus"
    with pytest.raises(ValueError, match="Corpus and index chunks differ"):
        lyric_mask_for_corpus(index, path)
