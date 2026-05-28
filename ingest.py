from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

documents = []

data_path = "data"

for file in os.listdir(data_path):

    path = os.path.join(data_path, file)

    if file.endswith(".txt"):
        loader = TextLoader(path)
        documents.extend(loader.load())

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
vectorstore = FAISS.from_documents(docs, embeddings)

# Save vector DB
vectorstore.save_local("vectorstore")

print("Vector database created successfully!")