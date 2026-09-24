# Natural text and stanza reranking: development experiment

66 development queries; historical observed choices are incomplete judgments.
The held-out split was not evaluated. Production defaults and index are unchanged.

| Configuration | Hit@5 | Recall@5 | nDCG@5 | Candidate Recall@50 | Warm p50 ms | Warm p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 13.6% | 8.3% | 6.06% | 25.0% | 850 | 1514 |
| natural_text | 16.7% | 9.8% | 7.03% | 26.5% | 798 | 1181 |
| title_plus_stanza | 15.2% | 8.3% | 6.08% | 25.0% | 954 | 1513 |
| both | 16.7% | 9.1% | 7.36% | 26.5% | 852 | 1311 |

Only embedding text and reranker chunk selection vary. Chunk boundaries,
title weight (2.5), lyric boost, top-50 candidates, and fixed fallback are unchanged.
Natural embeddings use whitespace cleanup on both queries and indexed chunks.
The lexical boost still uses its original preprocessing.

Candidate recall counts observed hymns available before reranking. Warm latency
includes query encoding, retrieval, and reranking; excludes startup/index building.
These are single sequential passes on the local device, not controlled load tests.
The two stanza variants should have identical candidate recall to their corresponding
embedding baseline. Raw rankings, timings, and artifact hashes are in the JSON report.

Run: `python -m experiments.ranking`
