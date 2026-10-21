"""
Agent mode: instead of searching once with the question as asked, Gemini gets a search tool and
decides what to search, in which corpus, and how many times, up to AGENT_MAX_SEARCHES, before
answering with [n] citations.

Each search runs the same hybrid search and reranker as the pipeline. Passages are numbered in the
order the agent first sees them, across all its searches, so [6] means the same passage everywhere
in the answer. The result has the pipeline's shape, {"answer", "docs"}, so ask() and cited_indices
work unchanged; it adds "searches" and "llm_calls".
"""
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from .chain import format_block
from .config import AGENT_MAX_SEARCHES, CORPORA

# Kept flush-left: indentation inside the string would be sent to the model on every line
AGENT_PROMPT = """\
You answer questions about the Monster Hunter game series using only passages returned by the search tool.
Search before answering. Write short, specific queries about one subject each: for a comparison, search for \
each monster or game separately; when one part of the question must be answered before the next can be looked \
up, search again using what you found.
Corpora: "mh_wiki_monsters" holds in-game monster details such as classification, elements, weaknesses and \
the game a monster debuted in; "wikipedia_pages" holds the games, release dates, sales, developers and consoles. \
Leave the corpus out to search both.
Passages are numbered across all your searches. Cite the ones you used as [1] or [2][3], placed after the \
statements they support, using the numbers exactly as given.
If the passages don't answer the question, say that you don't know and cite nothing.
Do not use Markdown formatting. Answer only the question asked, add extra information only if highly relevant.
"""

FINAL_INSTRUCTION = "The search limit is reached. Answer now from the passages above, citing them as instructed."
LIMIT_REACHED = "Search limit reached; this search was not run. Answer from the passages already found."


class PassageLog:
    """Every passage the agent has seen for one question, numbered in the order it first appeared."""

    def __init__(self):
        self.docs: list[Document] = []
        self._numbers: dict[str, int] = {}

    def add(self, docs: list[Document]) -> list[int]:
        """Numbers for these passages; a passage seen in an earlier search keeps its first number."""
        numbers = []
        for doc in docs:
            if doc.page_content not in self._numbers:
                self.docs.append(doc)
                self._numbers[doc.page_content] = len(self.docs)
            numbers.append(self._numbers[doc.page_content])
        return numbers


class Agent:
    """
    retrievers maps a corpus name, or None for all corpora, to a retriever that returns reranked
    passages: the compression retrievers the pipeline already builds. A corpus given here pins
    every search to it, whatever the model asks for.
    """

    def __init__(self, llm, retrievers: dict, max_searches: int = AGENT_MAX_SEARCHES, corpus: str | None = None):
        self.llm = llm
        self.retrievers = retrievers
        self.max_searches = max_searches
        self.corpus = corpus

    def invoke(self, question: str) -> dict:
        log = PassageLog()
        searches = []
        search = self._search_tool(log, searches)

        prompt = AGENT_PROMPT
        if self.corpus:
            prompt += f'Searches are limited to the "{self.corpus}" corpus.\n'
        messages = [SystemMessage(prompt), HumanMessage(question)]

        model = self.llm.bind_tools([search])
        llm_calls = 0
        while True:
            reply = model.invoke(messages)
            llm_calls += 1
            messages.append(reply)
            if not reply.tool_calls:
                break

            # Every tool call needs a reply message, including ones refused for the limit or bad arguments
            for call in reply.tool_calls:
                messages.append(ToolMessage(self._run(search, call, searches), tool_call_id=call["id"]))

            if len(searches) >= self.max_searches:
                messages.append(HumanMessage(FINAL_INSTRUCTION))
                # The tool stays declared, since the history contains calls to it, but can't be called again
                reply = self.llm.bind_tools([search], tool_choice="none").invoke(messages)
                llm_calls += 1
                break

        return {"answer": reply.text, "docs": log.docs, "searches": searches, "llm_calls": llm_calls}

    def _search_tool(self, log: PassageLog, searches: list):
        """A search tool bound to this question's passage log, so numbering carries across searches."""
        retrievers, pinned = self.retrievers, self.corpus

        @tool
        def search(query: str, corpus: Literal[CORPORA] | None = None) -> str:
            """Search the Monster Hunter corpus. Returns numbered passages to cite.

            Args:
                query: what to look up, about one subject, e.g. "Nargacuga debut game".
                corpus: "mh_wiki_monsters" or "wikipedia_pages"; leave out to search both.
            """
            corpus = pinned or corpus
            docs = retrievers[corpus].invoke(query)
            numbers = log.add(docs)
            searches.append({"query": query, "corpus": corpus, "passages": numbers})
            if not docs:
                return "No passages found."
            return "\n\n".join(format_block(n, log.docs[n - 1]) for n in numbers)

        return search

    def _run(self, search, call: dict, searches: list) -> str:
        if len(searches) >= self.max_searches:
            return LIMIT_REACHED
        try:
            return search.invoke(call["args"])
        except ValidationError as e:
            # e.g. a corpus that doesn't exist; tell the model so it can retry with valid arguments
            return f"Invalid search arguments: {e.errors()[0]['msg']}. Corpus must be one of {', '.join(CORPORA)}, or left out."
