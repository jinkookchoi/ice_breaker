from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is the purpose of model regularization?"),
]

res = chat.invoke(messages)
print(res)

for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)

chat.batch([messages])
