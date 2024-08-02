import asyncio

from langchain_ollama.chat_models import ChatOllama

if __name__ == "__main__":
    # local ollama app
    # llm = ChatOllama(model="llama3")

    # version 1
    # local ollama docker container
    # https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image
    # docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    # docker exec -it ollama ollama pull llama3
    llm = ChatOllama(base_url="localhost:11434", model="llama3")
    summary = llm.invoke(input="What is langchain?")
    print(summary)

    # version 2. stream
    async def langchain_stream():
        async for chunk in llm.astream(
            "Write me a 5 verse song about goldfish on the moon"
        ):
            yield str(chunk.content)

    async def main():
        async for chunk in langchain_stream():
            print(chunk, end="|")

    asyncio.run(main())
