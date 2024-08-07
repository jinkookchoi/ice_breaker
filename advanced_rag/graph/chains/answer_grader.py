from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableSequence, RunnableSerializable
from langchain_openai import ChatOpenAI


class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# 역할: 응답이 사용자의 질문을 적절히 해결했는지를 평가합니다.
# 특징:
#   - 해결 여부 평가: 응답이 질문에 대한 적절한 해결책을 제공하는지를 평가합니다.
#   - 이진 점수: 응답이 질문을 해결하는지를 'yes' 또는 'no'로 이진 평가합니다.
# 사용 예시: 사용자 질문에 대한 응답의 품질을 평가하여, 그 응답이 질문에 대해 충분히 설명하고 해결책을 제공했는지를 확인하는 데 사용됩니다.
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
