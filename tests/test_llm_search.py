import io
import json

import pytest

from experiments.llm_search import expand_query


def test_missing_key_fails_without_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        expand_query("King Joash")


def test_context_is_sent_and_original_query_is_retained(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")

    def urlopen(request, timeout):
        payload = json.loads(request.data)
        assert json.loads(payload["messages"][1]["content"]) == {
            "title": "King Joash", "context": "Repairing the temple"
        }
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "restoration worship"}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert expand_query("King Joash", context="Repairing the temple") == "King Joash. restoration worship"
