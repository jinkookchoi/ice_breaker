from langchain_openai import ChatOpenAI


def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]


tools = [add, multiply]

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

llm_with_tools = llm.bind_tools(tools)

# query = "What is 3 * 12? Also, what is 11 + 49?"
query = "What is 3 * 12?"

res = llm_with_tools.invoke(query)
print(res)

query = "What is 3 * 12? Also, what is 11 + 49?"

res2 = llm_with_tools.invoke(query).tool_calls  # type: ignore
print(res)
