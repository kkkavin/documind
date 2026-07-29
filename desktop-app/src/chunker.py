from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, length_function=len, separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_documents(docs)
    return chunks
    
