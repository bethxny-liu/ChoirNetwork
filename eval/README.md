# Evaluation datasets

Offline retrieval labels used by `python -m choirnetwork eval`.

## Primary dataset

[`datasets/service_hymns.csv`](datasets/service_hymns.csv) contains 106 sermon
titles and the two hymns manually observed in each service:

```csv
query,hymn_1_slug,hymn_2_slug,split,category,notes
Such Great Faith,136,208,development,observed_service,
```

- **66 development queries** may be used for analysis and configuration choices.
- **40 held-out test queries** must remain unevaluated until the system is frozen.
- Near-duplicate hope and devil-foothold queries are kept together in development.
- Hymn IDs are exact slugs validated against the 536-entry local corpus.

The two selected hymns are observed positives, not exhaustive relevance
judgments. Other hymns may also be appropriate for a sermon.

Every row includes its matched YouTube video ID, URL, and service date from the
Toronto church's public livestream archive. Video descriptions independently
confirm both hymn IDs for 89 services; 17 descriptions do not contain a
parseable hymn list. One manual transcription was corrected during this check:
the first hymn for "The Home of Noah" is 165, not 166.

Refresh the source fields with the reproducible metadata collector (requires
`yt-dlp`):

```bash
python eval/collect_youtube_sources.py
```

## Evaluation protocol

Use this process for the next benchmark:

1. Write the query list before running any retrieval system.
2. Assign each query to `development` or `test` before tuning. Do not move
   difficult queries after seeing results.
3. Have annotators select relevant hymns from the hymnal without viewing
   ChoirNetwork rankings. Record exact hymn slugs.
4. If two people label the data, reconcile disagreements while still blinded
   to model output.
5. Tune chunking, title weight, reranking, and expansion only on the
   development split.
6. Freeze code and labels before running the test split. Report both the split
   size and the command used.

Run one split with:

```bash
python -m choirnetwork eval --top-k 5 --split development
python -m choirnetwork eval --top-k 5 --split test --confirm-held-out
```

The current development command compares BM25, dense retrieval, dense + reranker,
dense + lyric boost, and dense + both. It writes JSON to `eval/results/current/`
with per-query rankings, labels, model names, and index/dataset checksums. Unknown
labels fail the run. The runner no longer depends on Bible grounding.

The original Bible-grounding ablation matrix and its reports remain historical
artifacts. The grounding code is retained under `experiments/`. Use frozen commit
`1ca1aff` to inspect the original runner. The original test results must not be
overwritten by a cleanup or used to select a new system.

The current frozen candidate is `dense_title_full`; development Bible
grounding was retained as a negative ablation and is disabled. After freezing
commit `1ca1aff`, the test command above was run once on September 7, 2026.
The candidate reached 7.2% nDCG@5 versus 5.1% for BM25. The complete output is
in [`results/service-test-bible.md`](results/service-test-bible.md).

## Metrics

See [`docs/ENGINE.md`](../docs/ENGINE.md#evaluation-and-interpretation) for Hit@k,
Recall@k, MRR@k, and nDCG@k definitions.

Development and held-out output are recorded under [`results/`](results/).
