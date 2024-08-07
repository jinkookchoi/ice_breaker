from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 역할: 검색된 문서가 사용자의 질문과 관련이 있는지를 평가합니다.
# 특징:
#   - 키워드 및 의미 평가: 문서가 질문과 관련된 키워드나 의미를 포함하고 있는지를 평가합니다.
#   - 이진 점수: 문서의 관련성 여부를 'yes' 또는 'no'로 이진 평가합니다.
# 사용 예시: 문서 검색 시스템에서 검색된 문서가 사용자의 질문에 얼마나 적합한지를 평가하는 데 사용됩니다.
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
