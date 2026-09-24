"""Development-only 2×2 comparison of natural text and lyric-stanza reranking."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from choirnetwork.engine import HymnIndex, HymnSimilarityEngine
from choirnetwork.eval import prepare_eval_queries, metrics_from_rankings, recall_at_k
from choirnetwork.preprocess import build_hymn_chunks, normalize_whitespace
from choirnetwork.scraper import load_hymns


class RankingExperiment(HymnSimilarityEngine):
    """Two experimental overrides; candidate selection and fallback stay unchanged."""

    def __init__(self, index, *, natural_text=False, stanza_only=False, lyric_mask=None):
        super().__init__(index)
        self.natural_text = natural_text
        self.stanza_only = stanza_only
        self.lyric_mask = lyric_mask

    def _encode_query(self, query):
        if not self.natural_text:
            return super()._encode_query(query)
        return self._get_model().encode(
            [normalize_whitespace(query)], convert_to_numpy=True, normalize_embeddings=True,
        )[0]

    def _best_chunk_for_hymn(self, hymn_idx, weighted_chunk_scores):
        if not self.stanza_only:
            return super()._best_chunk_for_hymn(hymn_idx, weighted_chunk_scores)
        indices = np.flatnonzero((self.index.chunk_hymn_indices == hymn_idx) & self.lyric_mask)
        if not len(indices):
            return self.index.titles[hymn_idx]
        best = indices[weighted_chunk_scores[indices].argmax()]
        return self.index.chunk_snippets[best]


def lyric_mask_for_corpus(index, corpus_path):
    """Identify stanza chunks from source structure, never from title text or weight."""
    hymns = load_hymns(corpus_path)
    chunks = [
        (i, chunk) for i, hymn in enumerate(hymns)
        for chunk in build_hymn_chunks(hymn.title, hymn.lyrics, title_weight=index.title_weight)
        if chunk.text
    ]
    if (
        [h.slug for h in hymns] != index.slugs
        or [c.snippet for _, c in chunks] != index.chunk_snippets
        or not np.array_equal([i for i, _ in chunks], index.chunk_hymn_indices)
    ):
        raise ValueError("Corpus and index chunks differ; use the corpus that built the baseline")
    return np.array([c.kind == "stanza" for _, c in chunks])


def run(index_path, corpus_path, dataset_path):
    index = HymnIndex.load(index_path)
    queries = prepare_eval_queries(index, dataset_path, split="development")
    lyric_mask = lyric_mask_for_corpus(index, corpus_path)
    baseline = HymnSimilarityEngine(index)
    model, cross = baseline._get_model(), baseline._get_cross_encoder()
    # Same chunk boundaries, title weight, model, boost, and fallback in every arm.
    start = perf_counter()
    natural = replace(index, chunk_embeddings=model.encode(
        [normalize_whitespace(text) for text in index.chunk_snippets],
        batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=True,
    ))
    build_seconds = perf_counter() - start
    results = []
    for natural_text, stanza_only in ((False, False), (True, False), (False, True), (True, True)):
        name = {
            (False, False): "baseline", (True, False): "natural_text",
            (False, True): "title_plus_stanza", (True, True): "both",
        }[natural_text, stanza_only]
        engine = RankingExperiment(
            natural if natural_text else index, natural_text=natural_text,
            stanza_only=stanza_only, lyric_mask=lyric_mask,
        )
        engine._model, engine._cross_encoder = model, cross
        engine.search(queries[0].query, top_k=5)  # Warm up before timing.
        rankings, candidate_recalls, timings = [], [], []
        for i, query in enumerate(queries, 1):
            start = perf_counter()
            matches = engine.search(query.query, top_k=5)
            timings.append((perf_counter() - start) * 1000)
            rankings.append([h.slug for h in matches])
            # The same engine without reranking exposes the recall-stage candidates.
            engine.use_reranker = False
            candidates = engine.search(query.query, top_k=index.recall_k)
            engine.use_reranker = True
            candidate_recalls.append(recall_at_k(
                [h.slug for h in candidates], set(query.relevant_slugs), index.recall_k,
            ))
            if i % 10 == 0:
                print(f"{name}: {i}/{len(queries)}", flush=True)
        results.append({
            "name": name,
            "metrics": metrics_from_rankings(queries, rankings, k=5),
            "candidate_recall": float(np.mean(candidate_recalls)),
            "warm_search_ms_p50": float(np.median(timings)),
            "warm_search_ms_p95": float(np.percentile(timings, 95)),
            "queries": [
                {"query": q.query, "observed": q.relevant_slugs, "top_5": ranking,
                 "candidate_recall": recall, "warm_search_ms": ms}
                for q, ranking, recall, ms in zip(queries, rankings, candidate_recalls, timings)
            ],
        })
        print(f"{name}: nDCG@5={results[-1]['metrics']['ndcg']:.5f}", flush=True)
    paths = [corpus_path, dataset_path, *(index_path / name for name in (
        "metadata.json", "chunk_embeddings.npy", "chunk_hymn_indices.npy", "chunk_weights.npy",
    ))]
    return {
        "split": "development", "k": 5, "candidate_k": index.recall_k,
        "model": index.model_name, "reranker": index.cross_encoder_model_name,
        "device": str(model.device), "title_weight": index.title_weight,
        "natural_index_build_seconds": build_seconds,
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "results": results,
    }


def markdown_report(report):
    lines = [
        "# Natural text and stanza reranking: development experiment", "",
        "66 development queries; historical observed choices are incomplete judgments.",
        "The held-out split was not evaluated. Production defaults and index are unchanged.", "",
        "| Configuration | Hit@5 | Recall@5 | nDCG@5 | Candidate Recall@50 | Warm p50 ms | Warm p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        m = result["metrics"]
        lines.append(
            f"| {result['name']} | {m['hit_rate']:.1%} | {m['recall']:.1%} | {m['ndcg']:.2%} "
            f"| {result['candidate_recall']:.1%} | {result['warm_search_ms_p50']:.0f} "
            f"| {result['warm_search_ms_p95']:.0f} |"
        )
    lines.extend([
        "", "Only embedding text and reranker chunk selection vary. Chunk boundaries,",
        "title weight (2.5), lyric boost, top-50 candidates, and fixed fallback are unchanged.",
        "Natural embeddings use whitespace cleanup on both queries and indexed chunks.",
        "The lexical boost still uses its original preprocessing.", "",
        "Candidate recall counts observed hymns available before reranking. Warm latency",
        "includes query encoding, retrieval, and reranking; excludes startup/index building.",
        "These are single sequential passes on the local device, not controlled load tests.",
        "The two stanza variants should have identical candidate recall to their corresponding",
        "embedding baseline. Raw rankings, timings, and artifact hashes are in the JSON report.", "",
        "Run: `python -m experiments.ranking`", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--corpus", type=Path, default=Path("data/raw/hymns.json"))
    parser.add_argument("--dataset", type=Path, default=Path("eval/datasets/service_hymns.csv"))
    parser.add_argument("--output", type=Path, default=Path("eval/results/natural-stanza-development"))
    args = parser.parse_args()
    report = run(args.index, args.corpus, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown_report(report))
    print(markdown_report(report))


if __name__ == "__main__":
    main()
