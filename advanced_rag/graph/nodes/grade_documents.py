import re
from typing import Any, Dict

from loguru import logger

from advanced_rag.graph.chains.retrieval_grader import retrieval_grader
from advanced_rag.graph.state import GraphState


def simplify_string(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    logger.info("--- STATE: CHECK DOCUMENT RELEVANCE TO QUESTION ---")
    question = state["question"]
    documents = state["documents"]

    logger.debug(question)

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        logger.debug(f"CHECK => {simplify_string(d.page_content)}")
        logger.debug(grade.lower())
        if grade.lower() == "yes":
            logger.info("--- GRADE: DOCUMENT RELEVANT (yes) ---")
            filtered_docs.append(d)
        else:
            logger.info("--- GRADE: DOCUMENT NOT RELEVANT (no) ---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
