"""Deterministic Bible-passage grounding for sermon-title queries."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from choirnetwork.lexical import BM25Index, DEFAULT_B, DEFAULT_K1
from choirnetwork.preprocess import preprocess_text

DEFAULT_BIBLE_PATH = (
    Path(__file__).resolve().parents[1] / "data/public/world_english_bible.json"
)
PASSAGE_WINDOW_SIZE = 5
PASSAGE_WINDOW_STRIDE = 3
BM25_K1 = DEFAULT_K1
BM25_B = DEFAULT_B
QUOTATION_MIN_TERMS = 4
QUOTATION_MIN_COVERAGE = 0.52
NARRATIVE_MIN_COVERAGE = 0.24
NARRATIVE_MIN_CONFIDENCE = 0.50
NARRATIVE_MIN_TERMS = 3
RARE_DOCUMENT_FREQUENCY = 12


@dataclass(frozen=True)
class BibleVerse:
    book: str
    chapter: int
    verse: str
    text: str


@dataclass(frozen=True)
class BiblePassage:
    book: str
    chapter: int
    start_verse: str
    end_verse: str
    text: str

    @property
    def reference(self) -> str:
        if self.start_verse == self.end_verse:
            return f"{self.book} {self.chapter}:{self.start_verse}"
        return (
            f"{self.book} {self.chapter}:"
            f"{self.start_verse}-{self.end_verse}"
        )


@dataclass(frozen=True)
class ParsedReference:
    book: str
    chapter: int
    start_verse: int | None = None
    end_verse: int | None = None


@dataclass(frozen=True)
class GroundingResult:
    original_query: str
    grounded_query: str
    reference: str | None
    passage_text: str
    source: str | None
    confidence: float
    query_type: str


BOOK_NAMES = [
    "Genesis",
    "Exodus",
    "Leviticus",
    "Numbers",
    "Deuteronomy",
    "Joshua",
    "Judges",
    "Ruth",
    "1 Samuel",
    "2 Samuel",
    "1 Kings",
    "2 Kings",
    "1 Chronicles",
    "2 Chronicles",
    "Ezra",
    "Nehemiah",
    "Esther",
    "Job",
    "Psalms",
    "Proverbs",
    "Ecclesiastes",
    "Song of Solomon",
    "Isaiah",
    "Jeremiah",
    "Lamentations",
    "Ezekiel",
    "Daniel",
    "Hosea",
    "Joel",
    "Amos",
    "Obadiah",
    "Jonah",
    "Micah",
    "Nahum",
    "Habakkuk",
    "Zephaniah",
    "Haggai",
    "Zechariah",
    "Malachi",
    "Matthew",
    "Mark",
    "Luke",
    "John",
    "Acts",
    "Romans",
    "1 Corinthians",
    "2 Corinthians",
    "Galatians",
    "Ephesians",
    "Philippians",
    "Colossians",
    "1 Thessalonians",
    "2 Thessalonians",
    "1 Timothy",
    "2 Timothy",
    "Titus",
    "Philemon",
    "Hebrews",
    "James",
    "1 Peter",
    "2 Peter",
    "1 John",
    "2 John",
    "3 John",
    "Jude",
    "Revelation",
]

_ABBREVIATIONS = {
    "gen": "Genesis",
    "exod": "Exodus",
    "exo": "Exodus",
    "lev": "Leviticus",
    "num": "Numbers",
    "deut": "Deuteronomy",
    "josh": "Joshua",
    "judg": "Judges",
    "ps": "Psalms",
    "psalm": "Psalms",
    "prov": "Proverbs",
    "eccl": "Ecclesiastes",
    "song of songs": "Song of Solomon",
    "songs": "Song of Solomon",
    "canticles": "Song of Solomon",
    "isa": "Isaiah",
    "jer": "Jeremiah",
    "lam": "Lamentations",
    "ezek": "Ezekiel",
    "dan": "Daniel",
    "hos": "Hosea",
    "obad": "Obadiah",
    "mic": "Micah",
    "nah": "Nahum",
    "hab": "Habakkuk",
    "zeph": "Zephaniah",
    "hag": "Haggai",
    "zech": "Zechariah",
    "mal": "Malachi",
    "matt": "Matthew",
    "mk": "Mark",
    "lk": "Luke",
    "jn": "John",
    "rom": "Romans",
    "gal": "Galatians",
    "eph": "Ephesians",
    "phil": "Philippians",
    "col": "Colossians",
    "thess": "Thessalonians",
    "tim": "Timothy",
    "tit": "Titus",
    "philem": "Philemon",
    "heb": "Hebrews",
    "jas": "James",
    "rev": "Revelation",
    "revelations": "Revelation",
}


def _book_aliases() -> dict[str, str]:
    aliases = {book.lower(): book for book in BOOK_NAMES}
    aliases.update(_ABBREVIATIONS)
    ordinal_words = {"1": "first", "2": "second", "3": "third"}
    for book in BOOK_NAMES:
        match = re.match(r"^([123]) (.+)$", book)
        if not match:
            continue
        number, base = match.groups()
        aliases[f"{ordinal_words[number]} {base.lower()}"] = book
        aliases[f"{number}{base.lower()}"] = book
        aliases[f"{number} {base[:3].lower()}"] = book
    return aliases


BOOK_ALIASES = _book_aliases()
_BOOK_PATTERN = "|".join(
    re.escape(alias) for alias in sorted(BOOK_ALIASES, key=len, reverse=True)
)
_REFERENCE_RE = re.compile(
    rf"\b(?P<book>{_BOOK_PATTERN})\.?\s+"
    r"(?P<chapter>\d{1,3})"
    r"(?:\s*:\s*(?P<start>\d{1,3})(?:\s*[-–]\s*(?P<end>\d{1,3}))?)?",
    re.IGNORECASE,
)


def parse_reference(query: str) -> ParsedReference | None:
    match = _REFERENCE_RE.search(query)
    if not match:
        return None
    book = BOOK_ALIASES[match.group("book").lower()]
    start = int(match.group("start")) if match.group("start") else None
    end = int(match.group("end")) if match.group("end") else start
    return ParsedReference(
        book=book,
        chapter=int(match.group("chapter")),
        start_verse=start,
        end_verse=end,
    )


def _stem_token(token: str) -> str:
    if token == "axe":
        return "ax"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        return token[:-1]
    return token


def _tokens(text: str, *, remove_stop_words: bool = True) -> list[str]:
    text = text.replace("’", "'")
    text = re.sub(r"\b(\w+)'s\b", r"\1", text)
    normalized = preprocess_text(text, remove_stop_words=remove_stop_words)
    return [_stem_token(token) for token in normalized.split()]


def _verse_number(value: str) -> int:
    match = re.match(r"\d+", value)
    return int(match.group()) if match else 0


class BibleIndex:
    """In-memory Bible lookup and BM25 index over overlapping passages."""

    def __init__(
        self,
        verses: list[BibleVerse],
        *,
        window_size: int = PASSAGE_WINDOW_SIZE,
        stride: int = PASSAGE_WINDOW_STRIDE,
    ):
        self.verses = verses
        self.window_size = window_size
        self.stride = stride
        grouped: dict[tuple[str, int], list[BibleVerse]] = defaultdict(list)
        for verse in verses:
            grouped[(verse.book, verse.chapter)].append(verse)
        self._chapters = dict(grouped)

        passages: list[BiblePassage] = []
        for chapter_verses in grouped.values():
            if len(chapter_verses) <= window_size:
                starts = [0]
            else:
                starts = list(range(0, len(chapter_verses), stride))
                final_start = max(0, len(chapter_verses) - window_size)
                if starts[-1] > final_start:
                    starts[-1] = final_start
                if final_start not in starts:
                    starts.append(final_start)
            for start in sorted(set(starts)):
                window = chapter_verses[start : start + window_size]
                if not window:
                    continue
                passages.append(
                    BiblePassage(
                        book=window[0].book,
                        chapter=window[0].chapter,
                        start_verse=window[0].verse,
                        end_verse=window[-1].verse,
                        text=" ".join(verse.text for verse in window),
                    )
                )
        self.passages = passages
        self._bm25 = BM25Index(
            [_tokens(passage.text) for passage in passages],
            k1=BM25_K1,
            b=BM25_B,
        )
        self.document_frequencies = self._bm25.document_frequencies
        self.inverse_document_frequencies = self._bm25.inverse_document_frequencies

    @classmethod
    def load(cls, path: Path = DEFAULT_BIBLE_PATH) -> BibleIndex:
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
        verses = [
            BibleVerse(
                book=item["book"],
                chapter=int(item["chapter"]),
                verse=str(item["verse"]),
                text=item["text"],
            )
            for item in payload["verses"]
        ]
        return cls(verses)

    def passage_for_reference(
        self, reference: ParsedReference
    ) -> BiblePassage | None:
        verses = self._chapters.get((reference.book, reference.chapter), [])
        if not verses:
            return None
        if reference.start_verse is None:
            selected = verses
        else:
            end = reference.end_verse or reference.start_verse
            selected = [
                verse
                for verse in verses
                if reference.start_verse <= _verse_number(verse.verse) <= end
            ]
        if not selected:
            return None
        return BiblePassage(
            book=reference.book,
            chapter=reference.chapter,
            start_verse=selected[0].verse,
            end_verse=selected[-1].verse,
            text=" ".join(verse.text for verse in selected),
        )

    def search(self, query: str, *, top_k: int = 1) -> list[tuple[BiblePassage, float]]:
        query_terms = _tokens(query)
        if not query_terms:
            return []
        scores = self._bm25.scores(query_terms)
        ranked = np.argsort(-scores, kind="stable")[:top_k]
        return [
            (self.passages[int(index)], float(scores[index]))
            for index in ranked
            if scores[index] > 0
        ]


def _weighted_coverage(query_terms: list[str], passage: BiblePassage, index: BibleIndex) -> float:
    unique_terms = set(query_terms)
    if not unique_terms:
        return 0.0
    passage_terms = set(_tokens(passage.text))
    weights = {
        term: index.inverse_document_frequencies.get(term, 0.0)
        for term in unique_terms
    }
    total = sum(weights.values())
    if total == 0:
        return 0.0
    return sum(weight for term, weight in weights.items() if term in passage_terms) / total


def _longest_phrase_ratio(query: str, passage: BiblePassage) -> float:
    query_terms = _tokens(query, remove_stop_words=False)
    passage_terms = _tokens(passage.text, remove_stop_words=False)
    if not query_terms:
        return 0.0
    previous = [0] * (len(passage_terms) + 1)
    longest = 0
    for query_term in query_terms:
        current = [0]
        for index, passage_term in enumerate(passage_terms, start=1):
            value = previous[index - 1] + 1 if query_term == passage_term else 0
            current.append(value)
            longest = max(longest, value)
        previous = current
    return longest / len(query_terms)


class BibleGrounder:
    def __init__(self, index: BibleIndex):
        self.index = index

    @classmethod
    def load(cls, path: Path = DEFAULT_BIBLE_PATH) -> BibleGrounder:
        return cls(BibleIndex.load(path))

    @lru_cache(maxsize=512)
    def ground(self, query: str) -> GroundingResult:
        query = query.strip()
        explicit = parse_reference(query)
        if explicit:
            passage = self.index.passage_for_reference(explicit)
            if passage:
                return self._result(
                    query,
                    passage,
                    source="explicit_reference",
                    confidence=1.0,
                    query_type="explicit_reference",
                )

        matches = self.index.search(query, top_k=1)
        if not matches:
            return self._abstract(query)
        passage, _ = matches[0]
        query_terms = _tokens(query)
        coverage = _weighted_coverage(query_terms, passage, self.index)
        phrase_ratio = _longest_phrase_ratio(query, passage)
        confidence = min(1.0, 0.75 * coverage + 0.25 * phrase_ratio)
        passage_terms = set(_tokens(passage.text))
        rare_matches = [
            term
            for term in set(query_terms)
            if self.index.document_frequencies.get(term, math.inf)
            <= RARE_DOCUMENT_FREQUENCY
            and term in passage_terms
        ]

        if (
            len(set(query_terms)) >= QUOTATION_MIN_TERMS
            and coverage >= QUOTATION_MIN_COVERAGE
            and (phrase_ratio >= 0.30 or rare_matches)
        ):
            return self._result(
                query,
                passage,
                source="passage_bm25",
                confidence=confidence,
                query_type="quotation",
            )

        if (
            len(set(query_terms)) >= NARRATIVE_MIN_TERMS
            and rare_matches
            and coverage >= NARRATIVE_MIN_COVERAGE
            and confidence >= NARRATIVE_MIN_CONFIDENCE
        ):
            return self._result(
                query,
                passage,
                source="passage_bm25",
                confidence=confidence,
                query_type="narrative",
            )
        return self._abstract(query)

    @staticmethod
    def _result(
        query: str,
        passage: BiblePassage,
        *,
        source: str,
        confidence: float,
        query_type: str,
    ) -> GroundingResult:
        return GroundingResult(
            original_query=query,
            grounded_query=(
                f"{query}. Bible context: {passage.reference}. {passage.text}"
            ),
            reference=passage.reference,
            passage_text=passage.text,
            source=source,
            confidence=confidence,
            query_type=query_type,
        )

    @staticmethod
    def _abstract(query: str) -> GroundingResult:
        return GroundingResult(
            original_query=query,
            grounded_query=query,
            reference=None,
            passage_text="",
            source=None,
            confidence=0.0,
            query_type="abstract",
        )
