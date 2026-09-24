import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from choirnetwork import api
from choirnetwork.engine import SimilarHymn


class FakeEngine:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [SimilarHymn(1, "", "1", "Grace", 0.5)]


@pytest.fixture
def client(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(api.app.state, "engine", engine, raising=False)
    return engine


def test_api_uses_default_search_without_expansion_or_percentages(client):
    engine = client
    response = api.search_hymns(api.SearchRequest(query=" Grace ", top_k=5))
    assert engine.calls == [("Grace", {"top_k": 5})]
    assert response.model_dump() == {"query": "Grace", "results": [
        {"label": "1", "title": "Grace", "url": "https://hymnal.tjc.org/hymnal-library/1", "snippet": ""}
    ]}


@pytest.mark.parametrize("payload,status", [
    ({"query": " "}, 400), ({"query": "x" * 301}, 422),
    ({"query": "Grace", "top_k": 0}, 422), ({"query": "Grace", "top_k": 51}, 422),
    ({"query": "Grace", "use_llm_expansion": True}, 422),
    ({"query": "Grace", "mode": "threshold"}, 422),
])
def test_invalid_or_removed_controls_are_rejected(client, payload, status):
    with pytest.raises((ValidationError, HTTPException)) as error:
        api.search_hymns(api.SearchRequest(**payload))
    if isinstance(error.value, HTTPException):
        assert error.value.status_code == status
    else:
        assert status == 422
    assert not client.calls
