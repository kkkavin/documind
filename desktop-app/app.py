import os
import streamlit as st
from src.loader import load_pdf
from src.chunker import split_documents
from src.vector_store import create_vector_store
from src.rag_chain import ask_question

st.set_page_config(page_title="DocuMind AI - Personal Document Assistant", page_icon="🧠")
st.title("🧠 DocuMind AI: Personalized Notes & Textbook QA")

with st.sidebar:
    st.header("Upload Materials")
    uploaded_file = st.file_uploader("Upload a PDF note or textbook", type=["pdf"])

    if uploaded_file and st.button("Process & Index PDF"):
        with st.spinner("Processing PDF and building vector index..."):
            os.makedirs("data", exist_ok=True)
            file_path = os.path.join("data", "uploaded_doc.pdf")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            docs = load_pdf("uploaded_doc")
            chunks = split_documents(docs)
            create_vector_store(chunks)
            st.success("✅ Document indexed successfully! Ready for questions.")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your uploaded notes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching notes & generating answer..."):
            response = ask_question(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})