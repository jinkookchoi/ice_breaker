from dotenv import load_dotenv
from langgraph.checkpoint import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from loguru import logger

from advanced_rag.graph.chains.answer_grader import answer_grader
from advanced_rag.graph.chains.hallucination_grader import hallucination_grader
from advanced_rag.graph.chains.router import RouteQuery, question_router
from advanced_rag.graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from advanced_rag.graph.nodes import generate, grade_documents, retrieve, web_search
from advanced_rag.graph.state import GraphState

load_dotenv()
memory = SqliteSaver.from_conn_string(":memory:")
memory = MemorySaver()


def decide_to_generate(state: GraphState) -> str:
    logger.info("--- STATE: ASSESS GRADED DOCUMENTS ---")

    if state["web_search"]:
        logger.info(
            "--- DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH ---"
        )
        return WEBSEARCH
    else:
        logger.info("--- DECISION: GENERATE ---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    logger.info("--- STATE: CHECK HALLUCINATIONS ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    logger.debug(score)

    if hallucination_grade := score.binary_score:
        logger.info("--- DECISION: GENERATION IS GROUNDED IN DOCUMENTS ---")
        logger.info("--- GRADE GENERATION vs QUESTION ---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        logger.debug(score)
        if answer_grade := score.binary_score:
            logger.info("--- DECISION: GENERATION ADDRESSES QUESTION ---")
            return "useful"
        else:
            logger.info("--- DECISION: GENERATION DOES NOT ADDRESS QUESTION ---")
            return "not useful"
    else:
        logger.info("--- DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY ---")
        return "not supported"


def route_question(state: GraphState) -> str:
    logger.info("--- STATE: ROUTE QUESTION ---")
    question = state["question"]
    logger.debug(question)
    logger.debug(question_router)
    source: RouteQuery = question_router.invoke({"question": question})
    logger.debug(source)
    if source.datasource == WEBSEARCH:
        logger.debug("--- ROUTE QUESTION => WEB SEARCH ---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        logger.debug("--- ROUTE QUESTION => RAG ---")
        return RETRIEVE


workflow = StateGraph(GraphState)
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)


app = workflow.compile(checkpointer=memory)

app.get_graph().draw_mermaid_png(output_file_path="advanced_rag_graph.png")
