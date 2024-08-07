from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 역할: LLM(대규모 언어 모델)이 생성한 응답이 검색된 사실에 기반하여 지지되는지를 평가합니다.
# 특징:
#   - 사실 기반 평가: 생성된 응답이 제공된 사실 집합에 근거하고 있는지를 평가합니다.
#   - 이진 점수: 응답이 사실에 기반하여 지지되는지를 'yes' 또는 'no'로 이진 평가합니다.
# 사용 예시: LLM이 생성한 응답이 신뢰할 수 있는지, 즉 제공된 사실에 근거하고 있는지를 확인하는 데 사용됩니다.
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
