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


class FakeAgent:
    """Answers like rag.agent.Agent: the pipeline's result plus the searches it made."""

    def __init__(self, docs):
        self.docs = docs
        self.questions = []

    def invoke(self, question):
        self.questions.append(question)
        searches = [{"query": "Rathalos weakness", "corpus": "mh_wiki_monsters", "passages": [1]}]
        return {"answer": "Dragon [1].", "docs": self.docs, "searches": searches, "llm_calls": 2}


def per_corpus(answerers):
    """One answerer serves every corpus filter; a dict gives each corpus its own."""
    if isinstance(answerers, dict):
        return answerers
    return {None: answerers, "wikipedia_pages": answerers, "mh_wiki_monsters": answerers}


@pytest.fixture
def client_with(monkeypatch):
    """A TestClient whose startup builds the given fakes instead of the real chains and agents."""
    def make(chains, agents=None):
        answerers = {"pipeline": per_corpus(chains), "agent": per_corpus(agents) if agents else {}}
        monkeypatch.setattr(api, "build_answerers", lambda: answerers)
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


def test_query_routes_to_the_requested_corpus(client_with):
    chains = {None: FakeChain([]), "wikipedia_pages": FakeChain([]), "mh_wiki_monsters": FakeChain([])}
    with client_with(chains) as client:
        client.post("/query", json={"question": "Who developed Monster Hunter Now?", "corpus": "wikipedia_pages"})
        client.post("/query", json={"question": "What is Rathalos weak to?"})

    assert chains["wikipedia_pages"].questions == ["Who developed Monster Hunter Now?"]
    assert chains[None].questions == ["What is Rathalos weak to?"]
    assert chains["mh_wiki_monsters"].questions == []


def test_agent_flag_routes_to_the_agent_and_returns_its_searches(client_with):
    doc = Document(page_content="text", metadata={"source": "Rathalos", "corpus": "mh_wiki_monsters"})
    chain, agent = FakeChain([doc]), FakeAgent([doc])
    with client_with(chain, agents=agent) as client:
        response = client.post("/query", json={"question": "What is Rathalos weak to?", "agent": True})

    assert agent.questions == ["What is Rathalos weak to?"]
    assert chain.questions == []
    body = response.json()
    assert body["searches"] == [{"query": "Rathalos weakness", "corpus": "mh_wiki_monsters", "passages": [1]}]
    assert body["sources"][0]["cited"] is True


def test_agent_uses_the_requested_corpus(client_with):
    agents = {None: FakeAgent([]), "wikipedia_pages": FakeAgent([]), "mh_wiki_monsters": FakeAgent([])}
    with client_with(FakeChain([]), agents=agents) as client:
        client.post("/query", json={"question": "Who made Monster Hunter Now?", "corpus": "wikipedia_pages", "agent": True})

    assert agents["wikipedia_pages"].questions == ["Who made Monster Hunter Now?"]
    assert agents[None].questions == []


def test_pipeline_response_has_no_searches(client_with):
    """Only agent mode chooses its own searches; the pipeline's response shape is unchanged."""
    with client_with(FakeChain([])) as client:
        assert "searches" not in client.post("/query", json={"question": "?"}).json()


def test_query_rejects_an_unknown_corpus(client_with):
    with client_with(FakeChain([])) as client:
        assert client.post("/query", json={"question": "?", "corpus": "fandom"}).status_code == 422


class FailingChain:
    def __init__(self, error):
        self.error = error

    def invoke(self, question):
        raise self.error


class ResourceExhausted(Exception):
    """Same name as the exception Google's client raises for a 429."""


@pytest.mark.parametrize("error", [
    RuntimeError("429 RESOURCE_EXHAUSTED. Quota exceeded for generate_content_free_tier_requests"),
    ResourceExhausted("Resource has been exhausted (e.g. check quota)."),
])
def test_rate_limit_gives_503_with_retry_advice(client_with, error):
    with client_with(FailingChain(error)) as client:
        response = client.post("/query", json={"question": "?"})
    assert response.status_code == 503
    assert "rate limiting" in response.json()["detail"]
    assert response.headers["Retry-After"] == "30"


def test_other_failures_give_503_without_leaking_the_error(client_with):
    """The provider's message can include request details; it belongs in the server log, not the response."""
    with client_with(FailingChain(ConnectionError("api.voyageai.com: key sk-abc123 refused"))) as client:
        response = client.post("/query", json={"question": "?"})
    assert response.status_code == 503
    assert "sk-abc123" not in response.text
    assert "unavailable" in response.json()["detail"]


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
