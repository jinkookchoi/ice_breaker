from typing import Dict, Any, List
from loguru import logger

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        logger.debug(f"***** Prompt to LLM was: *****\n\n{prompts[0]}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        logger.debug(f"$$$$$ LLM Response: $$$$$\n\n{response.generations[0][0].text}")
