from typing import Any, Dict

from loguru import logger

from advanced_rag.graph.state import GraphState
from advanced_rag.ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    logger.info("--- STATE: RETRIEVE ---")
    question = state["question"]

    documents = retriever.invoke(question)
    logger.debug(question)
    logger.debug(f"Retrieved documents counts={len(documents)}")
    return {"documents": documents, "question": question}
