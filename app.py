import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Page Title
st.title("🏛️ Telangana Tourism RAG Chatbot")

# Load Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=st.secrets["GROQ_API_KEY"]
)
)

# Load Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS Vector Store
db = FAISS.load_local(
    "vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)

# User Input
query = st.text_input("Ask about Telangana Tourism")

if query:
    # Retrieve relevant documents
    docs = db.similarity_search(query, k=3)

    # Create context
    context = "\n".join([doc.page_content for doc in docs])

    # Prompt
    prompt = f"""
    You are a Telangana Tourism Assistant.

    Answer the user's question only from the provided context.

    Context:
    {context}

    Question:
    {query}
    """

    # Get response
    response = llm.invoke(prompt)

    # Display answer
    st.subheader("Answer")
    st.write(response.content)
