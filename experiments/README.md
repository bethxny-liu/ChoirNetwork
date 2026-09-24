# Context experiments

These modules are outside the production API and the default evaluation matrix.

## LLM context expansion

```bash
python -m experiments.llm_search "King Joash" --context "Repairing the temple"
```

Set `OPENAI_API_KEY` in `.env`. The supplied title and context are sent to
OpenAI and the resulting themes are added to the retrieval query. The original
title is still sent to the reranker. It is an experiment, not part of the web
app or benchmark.

## Deterministic Bible grounding

`bible_grounding.py` retains the reference parser and passage-BM25 experiment.
The public-domain verse corpus is `data/public/world_english_bible.json`; its
builder is `eval/build_bible_corpus.py`. Programmatic use:

```python
from experiments.bible_grounding import BibleGrounder
result = BibleGrounder.load().ground("The crises of rebuilding (Ezra 4)")
print(result.reference, result.grounded_query)
```

It did not improve the selected development result. Its full historical ablation
reports remain in `eval/results/`. The original runner and curated/LLM expansion
implementation are available at frozen commit `1ca1aff`.

## Natural text and stanza reranking

Run `python -m experiments.ranking` to compare the baseline, natural-text
embeddings, title-plus-stanza reranking, and both changes using the development
split. The script uses the original corpus and index but does not change them.
Results are in [`eval/results/natural-stanza-development.md`](../eval/results/natural-stanza-development.md).

