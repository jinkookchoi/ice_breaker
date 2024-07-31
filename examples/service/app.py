# https://python.langchain.com/v0.2/docs/how_to/chat_streaming/

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from loguru import logger

app = FastAPI()
chat = ChatOpenAI()


async def langchain_stream():
    async for chunk in chat.astream(
        "Write me a 5 verse song about goldfish on the moon"
    ):
        logger.debug(chunk)
        yield str(chunk.content)


async def langchain_events():
    async for event in chat.astream_events(
        "Write me a 5 verse song about goldfish on the moon", version="v1"
    ):
        logger.debug(event)
        logger.debug(event["data"])
        if event["event"] == "on_chat_model_stream" and event["data"] is not None:
            if "chunk" in event["data"]:
                chunk = event["data"]["chunk"]
                content = chunk["content"]
                logger.debug(content)
                yield str(content)


@app.get("/stream")
async def stream():
    # return StreamingResponse(langchain_stream(), media_type="text/event-stream")
    return StreamingResponse(langchain_events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
