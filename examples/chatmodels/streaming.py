from langchain_openai.chat_models import ChatOpenAI

chat = ChatOpenAI()
for chunk in chat.stream("Write me a song about goldfish on the moon"):
    print(chunk.content, end="", flush=True)
