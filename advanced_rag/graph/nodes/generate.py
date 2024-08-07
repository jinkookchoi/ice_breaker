from typing import Any, Dict

from loguru import logger

from advanced_rag.graph.chains.generation import generation_chain
from advanced_rag.graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    logger.info("--- STATE: GENERATE ---")
    question = state["question"]
    documents = state["documents"]
    logger.debug(question)

    generation = generation_chain.invoke({"context": documents, "question": question})
    logger.success(generation)
    return {"documents": documents, "question": question, "generation": generation}
