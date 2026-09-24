"""Embedding and similarity retrieval for hymns."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from choirnetwork.preprocess import (
    DEFAULT_TITLE_WEIGHT,
    build_hymn_chunks,
    preprocess_text,
)
from choirnetwork.scraper import HymnRecord, hymn_label
from choirnetwork.theme_boost import extract_theme_keywords, lyric_keyword_boost

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RECALL_K = 50
INDEX_VERSION = 2
# Preserve the frozen benchmark's k=5 fallback, independently of requested k.
# This is a ranking heuristic, not a calibrated probability of relevance.
RERANK_MIN_RESULTS = 5
RERANK_MIN_SCORE = 0.005


@dataclass(frozen=True)
class SimilarHymn:
    number: int
    variant: str
    slug: str
    title: str
    score: float
    snippet: str = ""

    @property
    def label(self) -> str:
        return hymn_label(self.number, self.variant)


@dataclass
class HymnIndex:
    model_name: str
    numbers: list[int]
    variants: list[str]
    slugs: list[str]
    titles: list[str]
    index_version: int = INDEX_VERSION
    cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL
    title_weight: float = DEFAULT_TITLE_WEIGHT
    recall_k: int = DEFAULT_RECALL_K
    chunk_embeddings: np.ndarray | None = None
    chunk_hymn_indices: np.ndarray | None = None
    chunk_weights: np.ndarray | None = None
    chunk_snippets: list[str] | None = None
    lyrics_preprocessed: list[str] | None = None

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "index_version": self.index_version,
            "model_name": self.model_name,
            "cross_encoder_model_name": self.cross_encoder_model_name,
            "title_weight": self.title_weight,
            "recall_k": self.recall_k,
            "numbers": self.numbers,
            "variants": self.variants,
            "slugs": self.slugs,
            "titles": self.titles,
        }
        metadata["chunk_snippets"] = self.chunk_snippets
        metadata["lyrics_preprocessed"] = self.lyrics_preprocessed
        np.save(directory / "chunk_embeddings.npy", self.chunk_embeddings)
        np.save(directory / "chunk_hymn_indices.npy", self.chunk_hymn_indices)
        np.save(directory / "chunk_weights.npy", self.chunk_weights)

        with (directory / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory: Path) -> HymnIndex:
        with (directory / "metadata.json").open(encoding="utf-8") as file:
            metadata = json.load(file)

        if metadata.get("index_version") != INDEX_VERSION:
            raise ValueError("Unsupported index format. Run: python -m choirnetwork build")
        count = len(metadata["slugs"])
        lyrics = metadata.get("lyrics_preprocessed")
        if lyrics is None or len(lyrics) != count:
            raise ValueError("Index is missing lyrics. Run: python -m choirnetwork build")
        index = cls(
            **metadata,
            chunk_embeddings=np.load(directory / "chunk_embeddings.npy"),
            chunk_hymn_indices=np.load(directory / "chunk_hymn_indices.npy"),
            chunk_weights=np.load(directory / "chunk_weights.npy"),
        )
        chunks = len(index.chunk_embeddings)
        if not (
            len(index.numbers) == len(index.variants) == len(index.titles) == count
            and len(set(index.slugs)) == count
            and len(index.chunk_hymn_indices) == len(index.chunk_weights)
            == len(index.chunk_snippets) == chunks
            and chunks > 0
            and np.all((index.chunk_hymn_indices >= 0) & (index.chunk_hymn_indices < count))
        ):
            raise ValueError("Inconsistent index artifacts. Run: python -m choirnetwork build")
        return index

    def hymn_at(self, idx: int, score: float) -> SimilarHymn:
        return SimilarHymn(
            number=self.numbers[idx],
            variant=self.variants[idx],
            slug=self.slugs[idx],
            title=self.titles[idx],
            score=score,
        )


def _normalize_bi_encoder_score(raw_score: float, title_weight: float) -> float:
    """Map weighted max-pool cosine scores (0..title_weight) to 0..1."""
    if title_weight <= 0:
        return max(0.0, min(raw_score, 1.0))
    return max(0.0, min(raw_score / title_weight, 1.0))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


class HymnSimilarityEngine:
    def __init__(
        self,
        index: HymnIndex,
        *,
        use_reranker: bool = True,
        use_lyric_boost: bool = True,
    ):
        self.index = index
        self.use_reranker = use_reranker
        self.use_lyric_boost = use_lyric_boost
        self._model: SentenceTransformer | None = None
        self._cross_encoder: CrossEncoder | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.index.model_name)
        return self._model

    def _get_cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(self.index.cross_encoder_model_name)
        return self._cross_encoder

    def _encode_query(self, query: str) -> np.ndarray:
        text = preprocess_text(query, remove_stop_words=True)
        return self._get_model().encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

    @classmethod
    def from_raw_hymns(
        cls,
        hymns: list[HymnRecord],
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        title_weight: float = DEFAULT_TITLE_WEIGHT,
        recall_k: int = DEFAULT_RECALL_K,
    ) -> HymnSimilarityEngine:
        chunk_texts: list[str] = []
        chunk_hymn_indices: list[int] = []
        chunk_weights: list[float] = []
        chunk_snippets: list[str] = []

        for hymn_idx, hymn in enumerate(hymns):
            chunks = build_hymn_chunks(
                hymn.title,
                hymn.lyrics,
                title_weight=title_weight,
                remove_stop_words=True,
            )
            for chunk in chunks:
                if not chunk.text:
                    continue
                chunk_texts.append(chunk.text)
                chunk_hymn_indices.append(hymn_idx)
                chunk_weights.append(chunk.weight)
                chunk_snippets.append(chunk.snippet)

        model = SentenceTransformer(model_name)
        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        index = HymnIndex(
            model_name=model_name,
            numbers=[hymn.number for hymn in hymns],
            variants=[hymn.variant for hymn in hymns],
            slugs=[hymn.slug for hymn in hymns],
            titles=[hymn.title for hymn in hymns],
            index_version=INDEX_VERSION,
            cross_encoder_model_name=cross_encoder_model_name,
            title_weight=title_weight,
            recall_k=recall_k,
            chunk_embeddings=chunk_embeddings,
            chunk_hymn_indices=np.asarray(chunk_hymn_indices, dtype=np.int32),
            chunk_weights=np.asarray(chunk_weights, dtype=np.float32),
            chunk_snippets=chunk_snippets,
            lyrics_preprocessed=[
                preprocess_text(hymn.lyrics, remove_stop_words=False) for hymn in hymns
            ],
        )
        return cls(index)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        retrieval_query: str | None = None,
        rerank_query: str | None = None,
    ) -> list[SimilarHymn]:
        """Return a prefix of one ranking; optional query overrides support experiments."""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if not 1 <= top_k <= self.index.recall_k:
            raise ValueError(f"top_k must be between 1 and {self.index.recall_k}")
        search_query = retrieval_query or query
        hymn_scores, chunk_scores = self._chunk_scores(self._encode_query(search_query))
        if self.use_lyric_boost:
            hymn_scores = self._apply_lyric_boost(hymn_scores, search_query)
        candidates = [
            (int(idx), _normalize_bi_encoder_score(float(hymn_scores[idx]), self.index.title_weight))
            for idx in np.argsort(-hymn_scores)[:self.index.recall_k]
            if np.isfinite(hymn_scores[idx])
        ]
        ranked = self._rerank_candidates(rerank_query or query, candidates, chunk_scores)
        return [
            replace(match, snippet=self._best_lyric_stanza(
                self.index.slugs.index(match.slug), chunk_scores
            ))
            for match in ranked[:top_k]
        ]

    def _chunk_scores(self, query_embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.index.chunk_embeddings is not None
        assert self.index.chunk_hymn_indices is not None
        assert self.index.chunk_weights is not None

        scores = self.index.chunk_embeddings @ query_embedding
        weighted_scores = scores * self.index.chunk_weights

        hymn_scores = np.full(len(self.index.slugs), -np.inf, dtype=np.float32)
        np.maximum.at(hymn_scores, self.index.chunk_hymn_indices, weighted_scores)
        return hymn_scores, weighted_scores

    def _apply_lyric_boost(self, hymn_scores: np.ndarray, query: str) -> np.ndarray:
        if not self.index.lyrics_preprocessed:
            return hymn_scores

        keywords = extract_theme_keywords(query)
        if not keywords:
            return hymn_scores

        boosted = hymn_scores.copy()
        title_weight = self.index.title_weight
        for idx, lyrics in enumerate(self.index.lyrics_preprocessed):
            if not np.isfinite(boosted[idx]):
                continue
            boost = lyric_keyword_boost(lyrics, keywords)
            if boost > 0:
                boosted[idx] += boost * title_weight
        return boosted

    def _best_chunk_for_hymn(
        self,
        hymn_idx: int,
        weighted_chunk_scores: np.ndarray,
    ) -> str:
        assert self.index.chunk_hymn_indices is not None
        assert self.index.chunk_snippets is not None

        mask = self.index.chunk_hymn_indices == hymn_idx
        chunk_indices = np.flatnonzero(mask)
        if len(chunk_indices) == 0:
            return self.index.titles[hymn_idx]

        best_chunk_idx = chunk_indices[int(weighted_chunk_scores[chunk_indices].argmax())]
        return self.index.chunk_snippets[best_chunk_idx]

    def _best_lyric_stanza(
        self,
        hymn_idx: int,
        weighted_chunk_scores: np.ndarray,
    ) -> str:
        """Return the closest lyric stanza for display as match context."""
        assert self.index.chunk_hymn_indices is not None
        assert self.index.chunk_weights is not None
        assert self.index.chunk_snippets is not None

        mask = (self.index.chunk_hymn_indices == hymn_idx) & (
            self.index.chunk_weights < self.index.title_weight
        )
        chunk_indices = np.flatnonzero(mask)
        if len(chunk_indices) == 0:
            return ""
        best_idx = chunk_indices[int(weighted_chunk_scores[chunk_indices].argmax())]
        return self.index.chunk_snippets[best_idx]

    def _rerank_candidates(
        self,
        query: str,
        candidates: list[tuple[int, float]],
        weighted_chunk_scores: np.ndarray,
    ) -> list[SimilarHymn]:
        if not candidates:
            return []

        if not self.use_reranker:
            return [self.index.hymn_at(idx, score) for idx, score in candidates]

        pairs = []
        for hymn_idx, _ in candidates:
            snippet = self._best_chunk_for_hymn(hymn_idx, weighted_chunk_scores)
            pairs.append((query, f"{self.index.titles[hymn_idx]}. {snippet}"))

        cross_scores = self._get_cross_encoder().predict(pairs)
        reranked = sorted(
            zip(candidates, cross_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        accepted = [
            self.index.hymn_at(hymn_idx, _sigmoid(float(cross_score)))
            for (hymn_idx, _), cross_score in reranked
            if _sigmoid(float(cross_score)) >= RERANK_MIN_SCORE
        ]
        if len(accepted) >= RERANK_MIN_RESULTS:
            accepted_indices = {match.slug for match in accepted}
            dense_tail = [
                self.index.hymn_at(idx, score)
                for idx, score in candidates
                if self.index.slugs[idx] not in accepted_indices
            ]
            return accepted + dense_tail
        return [self.index.hymn_at(idx, score) for idx, score in candidates]


def load_engine(
    directory: Path,
    *,
    use_reranker: bool = True,
    use_lyric_boost: bool = True,
) -> HymnSimilarityEngine:
    index = HymnIndex.load(directory)
    return HymnSimilarityEngine(
        index,
        use_reranker=use_reranker,
        use_lyric_boost=use_lyric_boost,
    )
