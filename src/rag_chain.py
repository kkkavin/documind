import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.vector_store import query_vector_store

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("⚠️ GOOGLE_API_KEY not found in .env file! Please set it before running.")

def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page_label", "N/A")
        formatted.append(f"[Page {page}]:\n{doc.page_content}")
    return "\n\n".join(formatted)

def ask_question(question: str) -> str:
    docs = query_vector_store(question)

    context_text = format_docs(docs)

    prompt_template = """You are StudySync, an AI academic assistant. Answer the student's question based strictly on the provided context below. 

    Instructions:
    - Always state the page number(s) where you found the information.
    - If the answer cannot be found in the provided context, state clearly: "I could not find the answer to this question in the provided notes."
    - Keep your answer clear, accurate, and concise.

    Context:
    {context}

    Question: {question}

    Answer:"""

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    prompt = PromptTemplate.from_template(prompt_template)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"context": context_text, "question": question})
    return response