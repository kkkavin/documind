from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, length_function=len, separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_documents(docs)
    return chunks
    
if __name__ == "__main__":
    from loader import load_pdf
    chunks = split_documents(load_pdf('a'))
    if chunks:
        print("\n--- RAW TEXTchunks ---")
        print("Metadata:", chunks[0].metadata)
        print("Text Preview:\n", chunks[0].page_content)