from dotenv import load_dotenv

load_dotenv()
from loguru import logger

from advanced_rag.graph.graph import app

question1 = "What are the types of agent memory?"
# question1 = "김치의 핵심 재료는 뭐야?"

inputs = {"question": question1}

for output in app.stream(inputs, config={"configurable": {"thread_id": "2"}}):
    for key, value in output.items():
        logger.success(f"Finished running. key={key}:")
        if generated := value.get("generation"):
            logger.success(generated)
