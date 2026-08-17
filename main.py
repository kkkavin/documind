from src.loader import load_pdf
from src.chunker import split_documents
from src.vector_store import create_vector_store, query_vector_store
from src.rag_chain import ask_question

if __name__ == '__main__':
    raw_text = load_pdf('a')

    chunks = split_documents(raw_text)

    create_vector_store(chunks)

    while True:
        print('\n\n')
        query = input(">> ")
        if query.lower() == "exit":
            print("Bye!!")
            break
        
        response = ask_question(query)
        print()
        print(response)
