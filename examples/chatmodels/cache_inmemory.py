# https://python.langchain.com/v0.2/docs/how_to/chat_model_caching/

from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
set_llm_cache(InMemoryCache())

# The first time, it is not yet in cache, so it should take longer
res = llm.invoke("Tell me a joke")
print(res)

res = llm.invoke("Tell me a joke")
print(res)
