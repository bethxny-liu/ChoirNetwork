"""Tests for hymn corpus serialization."""

import json

import pytest

from choirnetwork.scraper import load_hymns


def test_load_hymns_reads_bible_verse_metadata(tmp_path):
    corpus_path = tmp_path / "hymns.json"
    corpus_path.write_text(
        json.dumps(
            {
                "hymns": [
                    {
                        "number": 1,
                        "variant": "",
                        "slug": "1",
                        "title": "Holy, Holy, Holy",
                        "lyrics": "Holy, holy, holy!",
                        "url": "https://example.com/1",
                        "bible_verse": "Holy, holy, holy is the Lord God Almighty.",
                        "bible_reference": "Rev 4:8",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    hymn = load_hymns(corpus_path)[0]

    assert hymn.bible_verse == "Holy, holy, holy is the Lord God Almighty."
    assert hymn.bible_reference == "Rev 4:8"


def test_load_hymns_accepts_legacy_corpus_without_bible_metadata(tmp_path):
    corpus_path = tmp_path / "hymns.json"
    corpus_path.write_text(
        json.dumps(
            {
                "hymns": [
                    {
                        "number": 1,
                        "variant": "",
                        "slug": "1",
                        "title": "Holy, Holy, Holy",
                        "lyrics": "Holy, holy, holy!",
                        "url": "https://example.com/1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    hymn = load_hymns(corpus_path)[0]

    assert hymn.bible_verse == ""
    assert hymn.bible_reference == ""


def test_scrape_cli_protects_reviewed_corpus_and_separate_outputs(monkeypatch, tmp_path):
    from choirnetwork.cli import main

    monkeypatch.setattr(
        "choirnetwork.cli.scrape_hymnal",
        lambda: (_ for _ in ()).throw(AssertionError("network call should not run")),
    )
    corpus = "data/raw/hymns.json"
    cases = [
        ["scrape", "--output", corpus],
        ["scrape", "--missing-output", corpus],
        ["scrape", "--output", str(tmp_path / "same.json"), "--missing-output", str(tmp_path / "same.json")],
    ]
    for args in cases:
        with pytest.raises(SystemExit, match="Choose separate output files"):
            main(args)
