from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from rag.agent import FINAL_INSTRUCTION, LIMIT_REACHED, Agent, PassageLog
from rag.chain import ask, cited_indices


def chunk(source, text, corpus="mh_wiki_monsters"):
    return Document(page_content=f"Source: {source}\n\n{text}", metadata={"source": source, "corpus": corpus})


MH_GAME = chunk("Monster_Hunter_(video_game)", "Monster Hunter is a 2004 game for the PlayStation 2.", "wikipedia_pages")
MH_SERIES = chunk("Monster_Hunter", "Monster Hunter is a series by Capcom.", "wikipedia_pages")
PS2 = chunk("PlayStation_2", "It was first released in Japan on March 4, 2000.", "wikipedia_pages")
PS2_SPECS = chunk("PlayStation_2", "The PS2 uses the Emotion Engine.", "wikipedia_pages")


def search_call(query, corpus=None, call_id=None):
    args = {"query": query} if corpus is None else {"query": query, "corpus": corpus}
    return {"name": "search", "args": args, "id": call_id or query}


def reply(text="", calls=()):
    return AIMessage(content=text, tool_calls=list(calls))


class FakeModel:
    """Replays scripted replies in order and records what it was sent, in place of Gemini."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.tool_choices = []

    def bind_tools(self, tools, tool_choice=None):
        self.tool_choices.append(tool_choice)
        return self

    def invoke(self, messages):
        self.sent.append(list(messages))
        return self.replies.pop(0)


class FakeRetriever:
    """Returns passages by query and records the queries it was asked."""

    def __init__(self, results=None):
        self.results = results or {}
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.results.get(query, [])


def retrievers(**results):
    """All three retrievers the agent can choose from, sharing one set of canned results."""
    return {name: FakeRetriever(results) for name in (None, "wikipedia_pages", "mh_wiki_monsters")}


def tool_messages(model, call=-1):
    return [m for m in model.sent[call] if isinstance(m, ToolMessage)]


# --------------------------------------------------------------------------
# PassageLog
# --------------------------------------------------------------------------

def test_passage_log_numbers_in_order_first_seen():
    log = PassageLog()
    assert log.add([MH_GAME, MH_SERIES]) == [1, 2]
    assert log.add([MH_SERIES, PS2]) == [2, 3]
    assert log.docs == [MH_GAME, MH_SERIES, PS2]


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

def test_two_searches_number_passages_across_searches():
    """The two-hop question the pipeline half-answers: the second search uses the first's result."""
    found = retrievers(**{"first Monster Hunter console": [MH_GAME, MH_SERIES],
                          "PlayStation 2 Japan launch": [PS2, PS2_SPECS]})
    model = FakeModel([
        reply(calls=[search_call("first Monster Hunter console")]),
        reply(calls=[search_call("PlayStation 2 Japan launch", corpus="wikipedia_pages")]),
        reply("It was the PlayStation 2 [1], which launched in Japan on March 4, 2000 [3]."),
    ])
    result = Agent(model, found).invoke("First console, and its Japan launch?")

    assert result["docs"] == [MH_GAME, MH_SERIES, PS2, PS2_SPECS]
    assert result["searches"] == [
        {"query": "first Monster Hunter console", "corpus": None, "passages": [1, 2]},
        {"query": "PlayStation 2 Japan launch", "corpus": "wikipedia_pages", "passages": [3, 4]},
    ]
    assert result["llm_calls"] == 3
    cited = [result["docs"][i - 1] for i in cited_indices(result["answer"], len(result["docs"]))]
    assert cited == [MH_GAME, PS2]


def test_second_search_results_continue_the_numbering():
    """What the model reads must use the same numbers it will cite."""
    found = retrievers(a=[MH_GAME], b=[PS2])
    model = FakeModel([reply(calls=[search_call("a")]), reply(calls=[search_call("b")]), reply("Done [2].")])
    Agent(model, found).invoke("?")

    second_result = tool_messages(model)[-1].content
    assert second_result.startswith("[2] Source: PlayStation_2\n")
    assert "Source: PlayStation_2\n\n" not in second_result, "the chunk's own source prefix is shown once"


def test_repeated_passage_keeps_its_first_number():
    found = retrievers(a=[MH_GAME, MH_SERIES], b=[MH_SERIES, PS2])
    model = FakeModel([reply(calls=[search_call("a")]), reply(calls=[search_call("b")]), reply("Done.")])
    result = Agent(model, found).invoke("?")

    assert result["searches"][1]["passages"] == [2, 3]
    assert len(result["docs"]) == 3
    assert tool_messages(model)[-1].content.startswith("[2] Source: Monster_Hunter\n")


def test_search_limit_forces_an_answer_without_the_tool():
    found = retrievers(a=[MH_GAME], b=[PS2])
    model = FakeModel([
        reply(calls=[search_call("a")]),
        reply(calls=[search_call("b")]),
        reply("Answer from what was found [1][2]."),
    ])
    result = Agent(model, found, max_searches=2).invoke("?")

    assert result["answer"] == "Answer from what was found [1][2]."
    assert result["llm_calls"] == 3
    assert model.tool_choices[-1] == "none"
    assert model.sent[-1][-1].content == FINAL_INSTRUCTION


def test_calls_beyond_the_limit_in_one_reply_are_refused():
    """Each tool call still gets a reply, as the model API requires, but only the first are run."""
    found = retrievers(a=[MH_GAME], b=[PS2], c=[PS2_SPECS])
    model = FakeModel([
        reply(calls=[search_call("a"), search_call("b"), search_call("c")]),
        reply("Done [1]."),
    ])
    result = Agent(model, found, max_searches=2).invoke("?")

    assert [s["query"] for s in result["searches"]] == ["a", "b"]
    assert found[None].queries == ["a", "b"]
    replies = tool_messages(model)
    assert len(replies) == 3
    assert replies[2].content == LIMIT_REACHED


def test_pinned_corpus_overrides_the_models_choice():
    found = retrievers(a=[MH_GAME])
    model = FakeModel([reply(calls=[search_call("a", corpus="mh_wiki_monsters")]), reply("Done [1].")])
    result = Agent(model, found, corpus="wikipedia_pages").invoke("?")

    assert found["wikipedia_pages"].queries == ["a"]
    assert found["mh_wiki_monsters"].queries == []
    assert result["searches"][0]["corpus"] == "wikipedia_pages"
    assert 'limited to the "wikipedia_pages" corpus' in model.sent[0][0].content


def test_invalid_corpus_is_reported_back_without_searching():
    found = retrievers(a=[MH_GAME])
    model = FakeModel([reply(calls=[search_call("a", corpus="fandom")]), reply("I don't know.")])
    result = Agent(model, found).invoke("?")

    assert result["searches"] == []
    assert all(r.queries == [] for r in found.values())
    error = tool_messages(model)[0].content
    assert error.startswith("Invalid search arguments")
    assert "wikipedia_pages" in error


def test_answer_without_searching_has_no_sources():
    model = FakeModel([reply("I don't know.")])
    result = Agent(model, retrievers()).invoke("What is the capital of Australia?")

    assert result == {"answer": "I don't know.", "docs": [], "searches": [], "llm_calls": 1}


def test_search_with_no_results_says_so():
    model = FakeModel([reply(calls=[search_call("nothing matches")]), reply("I don't know.")])
    Agent(model, retrievers()).invoke("?")

    assert tool_messages(model)[0].content == "No passages found."


def test_agent_works_with_the_cli_output():
    """ask() takes anything with invoke() returning answer and docs, so the CLI needs no agent-specific code."""
    found = retrievers(a=[MH_GAME, MH_SERIES], b=[PS2])
    model = FakeModel([reply(calls=[search_call("a")]), reply(calls=[search_call("b")]), reply("PS2 [1], March 4, 2000 [3].")])
    out = ask(Agent(model, found), "?")

    assert "[1] Monster_Hunter_(video_game)" in out
    assert "[3] PlayStation_2" in out
    assert "[2] Monster_Hunter " not in out, "uncited passages aren't listed"
