import csv
import json
import math

import pytest

from choirnetwork.bm25 import BM25Retriever
from choirnetwork.engine import HymnIndex
from choirnetwork.eval import (
    EvalQuery, prepare_eval_queries, hit_rate_at_k, recall_at_k,
    mrr_at_k, ndcg_at_k, metrics_from_rankings, write_evaluation_report,
)


def sample_index():
    return HymnIndex(
        model_name="unused", numbers=[1, 2, 3], variants=[""] * 3,
        slugs=["1", "2", "3"], titles=["Rock of Ages", "Amazing Grace", "Trust and Obey"],
        lyrics_preprocessed=["rock ages cleft me", "grace sweet sound", "trust obey"],
    )


def test_metrics_have_hand_calculated_values():
    relevant = {"2", "3"}
    retrieved = ["1", "3", "2"]
    assert hit_rate_at_k(retrieved, relevant, 3) == 1
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert mrr_at_k(retrieved, relevant, 3) == 0.5
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(expected)
    assert ndcg_at_k(["4"], relevant, 5) == 0


def test_bm25_lexical_match():
    assert BM25Retriever(sample_index()).search("sweet grace", top_k=1)[0].slug == "2"


def write_queries(path, test_label="3", test_split="test"):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["query", "hymn_1_slug", "hymn_2_slug", "split"])
        writer.writerow(["Grace", "1", "2", "development"])
        writer.writerow(["Trust", "2", test_label, test_split])


def test_split_filtering(tmp_path):
    path = tmp_path / "queries.csv"
    write_queries(path)
    assert [q.query for q in prepare_eval_queries(sample_index(), path, split="test")] == ["Trust"]


@pytest.mark.parametrize("label,split", [("999", "test"), ("", "test"), ("3", "typo")])
def test_bad_labels_or_splits_fail_instead_of_changing_denominator(tmp_path, label, split):
    path = tmp_path / "queries.csv"
    write_queries(path, label, split)
    with pytest.raises(ValueError):
        prepare_eval_queries(sample_index(), path, split="test")


def test_metrics_and_rankings_are_saved(tmp_path):
    queries = [EvalQuery("Grace", ("2",), "development")]
    rankings = [["2", "1"]]
    metrics = metrics_from_rankings(queries, rankings, k=2)
    assert metrics == {"queries_evaluated": 1, "hit_rate": 1, "recall": 1, "mrr": 1, "ndcg": 1}
    report = {"split": "development", "results": [{"metrics": metrics, "rankings": rankings}]}
    assert json.loads(write_evaluation_report(report, tmp_path).read_text()) == report
