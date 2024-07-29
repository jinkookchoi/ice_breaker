import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from loguru import logger

load_dotenv()


if __name__ == "__main__":
    logger.info("Start...")
    _current_path = os.path.abspath(os.path.dirname(__file__))
    doc_fn = os.path.join(_current_path, "mediumblog1.txt")

    loader = TextLoader(doc_fn)
    document = loader.load()

    logger.info("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    logger.info(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings()

    logger.info("Ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    logger.info("Finish...")
