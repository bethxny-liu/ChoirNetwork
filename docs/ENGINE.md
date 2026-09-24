# Search engine

## Offline indexing

1. Read the local hymn corpus.
2. Split each hymn into a title chunk and lyric stanzas, using four-line blocks
   when the source has no explicit stanza boundaries.
3. Lowercase, expand contractions, remove punctuation and stop words, and encode
   each chunk with `all-MiniLM-L6-v2`.
4. Save normalized embeddings, hymn membership, chunk text, and weights as JSON
   and NumPy arrays. Title weight is 2.5; stanza weight is 1.

The current index has 536 hymns and 3,469 384-dimensional embeddings. The loader
accepts only the current chunked format with embedded lyrics. It does not silently
load a legacy index or backfill lyrics from a different corpus.

## Online ranking

For a normalized query embedding, compute all chunk dot products:

```
hymn_score = max(chunk_cosine × chunk_weight)
```

Add a small boost for distinctive query terms found in lyrics. The term weight
is 0.08, capped at 0.35, scaled by title weight. Take the 50 highest-scoring hymns.
The cross-encoder scores the query paired with each hymn's title and its best
weighted chunk. This chunk can itself be the title; the frozen benchmark did
not guarantee a lyric stanza. Selecting a stanza separately is a future model
change requiring development evaluation.

The cross-encoder logits are converted with sigmoid; candidates below 0.005 are
dropped. If at least five remain, put them first in reranked order, followed by the
remaining candidates in dense order. Otherwise use the dense order throughout.
Return the requested prefix. The cutoff is a heuristic, not a calibrated
confidence score. Ranking stays consistent as `top_k` changes.

Dense scores are normalized once. Scores are internal; the UI and API expose
ranked recommendations without percentages or similarity thresholds.

## Context experiment

`experiments/llm_search.py` can add supplied sermon context to the retrieval
query. It is optional and not part of the benchmark or web app. See
[`experiments/README.md`](../experiments/README.md).

## Evaluation and interpretation

BM25 uses hymn titles and lyrics. Configuration names such as `dense_title`
mean the **query** contains only the sermon title; the indexed documents still
include lyrics. Development ablations isolate the lyric boost and reranker.

- Hit@k: fraction of queries with at least one observed hymn in the first k.
- Recall@k: fraction of the observed hymns retrieved, averaged over queries.
- MRR@k: reciprocal rank of the first observed hymn, or zero if absent.
- nDCG@k: binary relevance gains discounted by log2(rank + 1), normalized by
  the ideal ranking and averaged over queries.

Unknown hymn labels fail evaluation instead of shrinking the denominator. Reports
save rankings, exact labels, model names, and index/dataset SHA-256 checksums.
Historical reports remain unchanged. Model revisions and dependency versions were
not fully pinned in the original experiment, which limits exact reproduction in
a fresh environment.

The benchmark's two observed hymns per service are incomplete judgments. It
measures agreement with historical choices using the sermon title alone, not
whether every unselected hymn is irrelevant. The test set has only 40 queries.
