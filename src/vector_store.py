from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def create_vector_store(chunks, persist_directory="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    return vector_store


def query_vector_store(query_text, persist_directory="./chroma_db", k=3):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    return vector_store.similarity_search(query_text, k=k)