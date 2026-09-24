"""Evaluate historical service selections; missing labels are errors, not exclusions."""

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from choirnetwork.bm25 import BM25Retriever
from choirnetwork.engine import HymnIndex, load_engine


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_slugs: tuple[str, ...]
    split: str


# Names match the original report: "title" means a sermon-title query,
# not an index containing only hymn titles.
RETRIEVAL_CONFIGS = (
    ("dense_title", False, False),
    ("dense_title_rerank", True, False),
    ("dense_title_boost", False, True),
    ("dense_title_full", True, True),
)


def prepare_eval_queries(index: HymnIndex, path: Path, *, split: str) -> list[EvalQuery]:
    if split not in {"development", "test"}:
        raise ValueError("Expected development or test split")
    queries = []
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["split"] not in {"development", "test"}:
                raise ValueError(f"Unknown split for {row['query']!r}")
            if row["split"] != split:
                continue
            labels = tuple(sorted({row["hymn_1_slug"].strip().lower(), row["hymn_2_slug"].strip().lower()}))
            missing = set(labels) - set(index.slugs)
            if missing:
                raise ValueError(f"Unknown hymn labels for {row['query']!r}: {sorted(missing)}")
            query = row["query"].strip()
            if not query:
                raise ValueError("Evaluation query cannot be empty")
            queries.append(EvalQuery(query, labels, split))
    if not queries:
        raise ValueError(f"No queries in split {split!r}")
    return queries


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if relevant.intersection(retrieved[:k]) else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved[:k])) / len(relevant)


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, slug in enumerate(retrieved[:k], start=1):
        if slug in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    def dcg(scores: list[float]) -> float:
        total = 0.0
        for index, score in enumerate(scores):
            total += score / math.log2(index + 2)
        return total

    gains = [1.0 if slug in relevant else 0.0 for slug in retrieved[:k]]
    ideal_gains = [1.0] * min(len(relevant), k)
    ideal_gains.extend([0.0] * max(0, k - len(ideal_gains)))

    ideal_dcg = dcg(ideal_gains)
    if ideal_dcg == 0:
        return 0.0
    return dcg(gains) / ideal_dcg


def metrics_from_rankings(queries: list[EvalQuery], rankings: list[list[str]], *, k: int) -> dict:
    if k < 1 or not queries or len(queries) != len(rankings):
        raise ValueError("Need positive k and exactly one ranking per query")
    metrics = {"queries_evaluated": len(queries)}
    for name, metric in (("hit_rate", hit_rate_at_k), ("recall", recall_at_k), ("mrr", mrr_at_k), ("ndcg", ndcg_at_k)):
        metrics[name] = sum(
            metric(ranking, set(query.relevant_slugs), k)
            for query, ranking in zip(queries, rankings)
        ) / len(queries)
    return metrics


def compare_retrieval_configs(index_path: Path, eval_path: Path, *, k: int = 5, split: str = "development") -> dict:
    engine = load_engine(index_path)
    queries = prepare_eval_queries(engine.index, eval_path, split=split)
    baseline = BM25Retriever(engine.index)
    results = []
    configurations = [("bm25_title", False, False), *RETRIEVAL_CONFIGS]
    for name, rerank, boost in configurations:
        engine.use_reranker, engine.use_lyric_boost = rerank, boost
        retriever = baseline if name == "bm25_title" else engine
        rankings = [[hymn.slug for hymn in retriever.search(query.query, top_k=k)] for query in queries]
        results.append({"name": name, "metrics": metrics_from_rankings(queries, rankings, k=k), "rankings": rankings})
    artifacts = [eval_path, *(index_path / name for name in (
        "metadata.json", "chunk_embeddings.npy", "chunk_hymn_indices.npy", "chunk_weights.npy"
    ))]
    return {
        "split": split,
        "k": k,
        "ranking_policy": "frozen-top-five-fallback-v1",
        "model_name": engine.index.model_name,
        "cross_encoder_model_name": engine.index.cross_encoder_model_name,
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},
        "queries": [asdict(query) for query in queries],
        "results": results,
    }


def write_evaluation_report(report: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"service-{report['split']}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
