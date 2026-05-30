import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.environ["GROQ_API_KEY"]
)

st.title("Telangana Tourism RAG Chatbot")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)

llm = OllamaLLM(model="llama3")

query = st.text_input("Ask about Telangana Tourism")

if query:
    docs = db.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    st.write(response)
