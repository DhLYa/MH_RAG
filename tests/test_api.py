import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import rag.api as api


class FakeChain:
    def __init__(self, docs):
        self.docs = docs
        self.questions = []

    def invoke(self, question):
        self.questions.append(question)
        return {"answer": "Rathalos is weak to Dragon [1].", "docs": self.docs}


@pytest.fixture
def client_with(monkeypatch):
    """A TestClient whose startup builds the given fake chain instead of the real one."""
    def make(chain):
        monkeypatch.setattr(api, "build_rag_chain", lambda: chain)
        return TestClient(api.app)
    return make


def test_health(client_with):
    with client_with(FakeChain([])) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_query_returns_answer_and_numbered_sources(client_with):
    cited = Document(
        page_content="text",
        metadata={"source": "Rathalos", "corpus": "mh_wiki_monsters", "relevance_score": 0.81},
    )
    uncited = Document(
        page_content="text",
        metadata={"source": "Rathian", "corpus": "mh_wiki_monsters", "relevance_score": 0.42},
    )
    chain = FakeChain([cited, uncited])
    with client_with(chain) as client:
        response = client.post("/query", json={"question": "What is Rathalos weak to?"})

    assert response.status_code == 200
    assert chain.questions == ["What is Rathalos weak to?"]
    assert response.json() == {
        "answer": "Rathalos is weak to Dragon [1].",
        "sources": [
            {"id": 1, "title": "Rathalos", "corpus": "mh_wiki_monsters", "relevance": 0.81, "cited": True},
            {"id": 2, "title": "Rathian", "corpus": "mh_wiki_monsters", "relevance": 0.42, "cited": False},
        ],
    }


def test_query_with_no_sources(client_with):
    with client_with(FakeChain([])) as client:
        assert client.post("/query", json={"question": "?"}).json()["sources"] == []


def test_query_rejects_a_missing_question(client_with):
    with client_with(FakeChain([])) as client:
        assert client.post("/query", json={}).status_code == 422


def test_startup_fails_on_an_empty_store(monkeypatch):
    """Without this the API would start and answer "I don't know" to everything."""
    monkeypatch.setattr(api, "get_vector_store", lambda: object())
    monkeypatch.setattr(api, "is_empty", lambda store: True)
    with pytest.raises(RuntimeError, match="Vector store is empty"):
        with TestClient(api.app):
            pass
