import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from readthedoc_search.consts import INDEX_NAME

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs() -> None:
    # wget -r -A.html -P docs https://api.python.langchain.com/en/latest/langchain_api_reference.html
    doc_fn = "docs/api.python.langchain.com/en/latest"

    logger.info(doc_fn)
    loader = ReadTheDocsLoader(doc_fn)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("docs", "https:/")
        doc.metadata.update({"source": new_url})

    logger.info(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    logger.success("Loading to vectorstore done")


if __name__ == "__main__":
    ingest_docs()
