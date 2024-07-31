# We can do the same thing with a SQLite cache
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

llm = ChatOpenAI(model="gpt-4")

# The first time, it is not yet in cache, so it should take longer
res = llm.invoke("Tell me a joke")
print(res)

res = llm.invoke("Tell me a joke")
print(res)
