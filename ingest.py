from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import embeddings

print("Loading PDF...")

loader = PyPDFLoader("data/demo.pdf")
documents = loader.load()

print(documents)
print(f"Loaded {len(documents)} pages")

print("Splitting document...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   #500 characters per chunk
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(chunks)

print(f"Created {len(chunks)} chunks")

print("Creating vector database...")

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Done!")