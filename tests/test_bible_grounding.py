"""Tests for deterministic sermon-title Bible grounding."""

from experiments.bible_grounding import (
    BibleGrounder,
    BibleIndex,
    BibleVerse,
    parse_reference,
)


def _sample_grounder() -> BibleGrounder:
    verses = [
        BibleVerse(
            "Genesis",
            15,
            "6",
            "He believed in Yahweh, who credited it to him for righteousness.",
        ),
        BibleVerse(
            "Ruth",
            1,
            "16",
            "Ruth said, Do not urge me to leave you or return from following you.",
        ),
        BibleVerse(
            "Ruth",
            1,
            "17",
            "Where you die, I will die, and there I will be buried.",
        ),
        BibleVerse(
            "Ezra",
            4,
            "1",
            "The adversaries heard that the children of the captivity were building.",
        ),
        BibleVerse(
            "Ezra",
            4,
            "2",
            "They came near and asked to build with them.",
        ),
        BibleVerse(
            "1 Samuel",
            24,
            "6",
            "I will not stretch out my hand against my lord, the anointed.",
        ),
        BibleVerse(
            "1 Samuel",
            24,
            "7",
            "David restrained his men and did not allow them to rise against Saul.",
        ),
    ]
    return BibleGrounder(BibleIndex(verses))


def test_parse_reference_supports_book_aliases_and_ranges():
    parsed = parse_reference("Do not harm the anointed (First Samuel 24:6-7)")

    assert parsed is not None
    assert parsed.book == "1 Samuel"
    assert parsed.chapter == 24
    assert parsed.start_verse == 6
    assert parsed.end_verse == 7


def test_explicit_chapter_reference_returns_the_chapter():
    result = _sample_grounder().ground("The crises of rebuilding (Ezra 4)")

    assert result.query_type == "explicit_reference"
    assert result.reference == "Ezra 4:1-2"
    assert "adversaries" in result.grounded_query


def test_strong_phrase_overlap_is_categorized_as_quotation():
    result = _sample_grounder().ground(
        "He believed in Yahweh who credited it to him for righteousness"
    )

    assert result.query_type == "quotation"
    assert result.reference == "Genesis 15:6"


def test_distinctive_bible_names_are_categorized_as_narrative():
    result = _sample_grounder().ground("Naomi returns with Ruth")

    assert result.query_type == "narrative"
    assert result.reference == "Ruth 1:16-17"


def test_uncertain_abstract_title_abstains():
    result = _sample_grounder().ground("A joyful life")

    assert result.query_type == "abstract"
    assert result.reference is None
    assert result.grounded_query == "A joyful life"


def test_grounding_is_deterministic():
    grounder = _sample_grounder()

    assert grounder.ground("Naomi returns with Ruth") == grounder.ground(
        "Naomi returns with Ruth"
    )
