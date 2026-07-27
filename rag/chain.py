from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
 
from .config import LLM_MODEL

def get_llm():
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    return llm

def format_docs(docs: list[Document]) -> str:
    doc_blocks = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source")
        doc_blocks.append(f"[{i}] Source: {source}\n{d.page_content}")
    return "\n\n".join(doc_blocks)

def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template("""
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, say that you don't know.
    Reformat the answer as appropriate and do not use Markdown formatting.
    Answer only the question asked, add extra information only if highly relevant.
    Answers should be strictly gameplay related unless asked otherwise. 
    Context: {context}
    Question: {question}
    """)

def build_chain(retriever) -> Runnable:
    answer = (
        {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | build_prompt()
        | get_llm()
        | StrOutputParser()
    )
    return RunnableParallel(
        docs=retriever,
        question=RunnablePassthrough(), 
    ) | RunnableParallel(
        answer=answer,
        docs=lambda x: x["docs"],
    )

def ask(chain, question: str) -> str:
    result = chain.invoke(question)
    docs = result["docs"]

    width = max(len(str(d.metadata.get("source"))) for d in docs)

    lines = [result["answer"], "\nSources:"]
    for i, d in enumerate(result["docs"], 1):
        score = d.metadata.get("relevance_score")
        score = f"{score:.4f}"
        lines.append(f"  [{i}] {d.metadata.get('source'):{width}}   relevance={score}")
    return "\n".join(lines)
