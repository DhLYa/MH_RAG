import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import LLM_MODEL

# Kept flush-left: indentation inside the string would be sent to the model on every line
PROMPT_TEMPLATE = """\
Use the numbered context blocks below to answer the question at the end.
If you don't know the answer, say that you don't know and cite nothing.
Cite the blocks you used as [1] or [2][3], placed after the statements they support.
Reformat the answer as appropriate and do not use Markdown formatting.
Answer only the question asked, add extra information only if highly relevant.
Context:
{context}
Question: {question}
"""

# "[2]", "[2][3]" and "[2, 3]" all cite blocks 2 and 3
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

def get_llm():
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)#, temperature=LLM_TEMPERATURE)
    return llm

def format_block(number: int, doc: Document) -> str:
    """One numbered context block, e.g. "[3] Source: Rathalos" followed by the passage."""
    source = doc.metadata.get("source")
    # Chunks already open with "Source: <title>" for retrieval; the block header says it once
    content = doc.page_content.removeprefix(f"Source: {source}\n\n")
    return f"[{number}] Source: {source}\n{content}"

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(format_block(i, d) for i, d in enumerate(docs, 1))

def cited_indices(answer: str, n_docs: int) -> list[int]:
    """Block numbers cited in the answer, in order of first citation. Numbers outside
    1..n_docs are ignored, since they don't refer to a block the model was given."""
    cited = []
    for group in CITATION.findall(answer):
        for number in group.split(","):
            i = int(number)
            if 1 <= i <= n_docs and i not in cited:
                cited.append(i)
    return cited

def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def build_answer_chain() -> Runnable:
    """Answers from documents already retrieved: {"docs": [...], "question": str} -> str."""
    return (
        {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | build_prompt()
        | get_llm()
        | StrOutputParser()
    )

def build_chain(retriever) -> Runnable:
    return RunnableParallel(
        docs=retriever,
        question=RunnablePassthrough(),
    ) | RunnableParallel(
        answer=build_answer_chain(),
        docs=lambda x: x["docs"],
    )

def describe_searches(searches: list[dict]) -> list[str]:
    """Agent mode's searches, one per line, e.g. '  1. "Tigrex debut" (mh_wiki_monsters)'."""
    lines = ["Searched:"]
    for i, search in enumerate(searches, 1):
        where = search["corpus"] or "all corpora"
        lines.append(f'  {i}. "{search["query"]}" ({where})')
    return lines

def ask(chain, question: str) -> str:
    """Runs the pipeline chain or the agent and formats the answer with its sources for the CLI."""
    result = chain.invoke(question)
    docs = result["docs"]

    lines = []
    # Only the agent reports searches; listing them shows where the passage numbers came from
    if result.get("searches"):
        lines += describe_searches(result["searches"]) + [""]
    lines.append(result["answer"])

    if not docs:
        lines.append("\nSources: none retrieved")
        return "\n".join(lines)

    # Show what the answer cites, numbered as in the answer; fall back to everything retrieved
    cited = cited_indices(result["answer"], len(docs))
    if cited:
        heading, shown = "\nSources:", sorted(cited)
    else:
        heading, shown = "\nRetrieved (none cited):", range(1, len(docs) + 1)

    width = max(len(str(docs[i - 1].metadata.get("source"))) for i in shown)

    lines.append(heading)
    for i in shown:
        d = docs[i - 1]
        score = d.metadata.get("relevance_score")
        score = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        source = str(d.metadata.get("source"))
        lines.append(f"  [{i}] {source:{width}}   relevance={score} - {d.metadata.get('corpus')}")
    return "\n".join(lines)
