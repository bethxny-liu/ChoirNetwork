# ChoirNetwork

Recommend hymns from the 536-entry True Jesus Church English hymnal using a
sermon title. Python, Sentence Transformers/PyTorch, NumPy, and FastAPI power
one local index and a small web interface.

```
Sermon title → MiniLM similarity over 3,469 title/stanza chunks
             → best weighted chunk per hymn + lyric keyword boost
             → top 50 hymns → cross-encoder reranking → recommendations
```

Embeddings are computed offline. A NumPy dot product is enough for this corpus;
there is no vector database or model training. The bi-encoder is
`all-MiniLM-L6-v2`; the cross-encoder is `ms-marco-MiniLM-L-6-v2`.

The cross-encoder was trained for web search, so the system falls back to dense
retrieval when fewer than five reranked candidates pass its score cutoff. The UI
shows rank order; model scores are not calibrated relevance probabilities.

## Run

Use Python 3.10+ for a new environment, then install `requirements.txt`.

```bash
pip install -r requirements.txt
python -m choirnetwork build     # requires the reviewed data/raw/hymns.json
python -m choirnetwork serve
python -m choirnetwork search "Holy, Holy, Holy" --top-k 10
python -m pytest tests/
```

The API and CLI use the same default search configuration. Production search
makes no LLM calls and does not read an OpenAI API key.

## Corpus and ingestion

The reviewed local corpus and index contain **536 hymns and 3,469 chunks**.
Lyrics for 497 entries came from hymnal.tjc.org. The remaining 39 were
reconstructed from positioned syllables in the English hymnbook PDF and are
marked as unreviewed transcriptions. Titles and Bible-reference metadata were
checked against the PDF. Those 39 transcriptions still need human proofreading.

The copyrighted hymn corpus and PDF are excluded from Git. A fresh clone does
not contain the complete benchmark corpus, and scraping alone cannot recreate
its PDF corrections. To collect available website lyrics:

```bash
python -m choirnetwork scrape
# Writes data/raw/scraped_hymns.json and data/raw/missing_lyrics.json.
# Review and merge this staging data into your local corpus before building.
```

Scraping does not overwrite the reviewed corpus. Keep that corpus and its index
together when reproducing results. Old single-vector indexes are no longer
supported; rebuild with `build`.

## Evaluation

The dataset records two hymns chosen in each of **106 historical services**:
66 development queries and 40 held-out test queries. These are observed positive
examples, not an exhaustive list of relevant hymns. Passage interpretation,
congregational familiarity, service order, and other considerations may influence
selection. A different recommendation is not necessarily inappropriate.

The configuration was selected on development data before the September 7, 2026
held-out run at commit `1ca1aff`:

| Configuration | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 10.0% | 6.2% | 6.1% | 5.1% |
| Dense + lyric boost + reranking/fallback | 15.0% | 8.8% | 8.5% | 7.2% |

This is **40.7% relative improvement in nDCG@5**, rounded to 41%, or about
2.1 percentage points. It is a small benchmark with incomplete judgments;
statistical significance has not been established. The historical reports remain
in `eval/results/`; the 41% claim refers to that frozen experiment.

```bash
python -m choirnetwork eval --split development
```

New reports go to `eval/results/current/` and include per-query rankings and
index/dataset checksums. Test evaluation requires `--split test
--confirm-held-out`; do not use repeated test runs to choose configurations.
See [the protocol](eval/README.md).

## Context experiments

Additional sermon context may help distinguish ambiguous titles. The current
benchmark is too small and its labels too incomplete to settle that question.
LLM expansion remains available as a separate, opt-in experiment:

```bash
cp .env.example .env  # add your OPENAI_API_KEY
python -m experiments.llm_search "King Joash" --context "Repairing the temple"
```

This sends the supplied title and context to OpenAI, prints the expanded query,
and uses it for dense recall. Reranking still uses the original title to isolate
the expansion experiment. It does not run in the web app or the benchmark and
is not covered by the 41% claim. See [experiments](experiments/README.md) for
limitations and the retained Bible-grounding experiment.

## Code map

- `choirnetwork/engine.py`: index I/O, embedding, retrieval, reranking
- `choirnetwork/preprocess.py`: text cleanup and title/stanza chunks
- `choirnetwork/theme_boost.py`: small lyric keyword heuristic
- `choirnetwork/bm25.py`, `lexical.py`: baseline and shared BM25 scorer
- `choirnetwork/scraper.py`: website ingestion and corpus records
- `choirnetwork/eval.py`: labels, metrics, ablations, reports
- `choirnetwork/api.py`, `cli.py`, `static/`: thin interfaces
- `experiments/`: optional context research, outside the production pipeline

[Engine details](docs/ENGINE.md) explain the ranking policy and tradeoffs.

Contributors: Bethany Liu, Mark Chen
