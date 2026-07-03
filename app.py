import streamlit as st

from langchain_chroma import Chroma

from config import embeddings, llm

st.set_page_config(page_title="Simple RAG")

st.title("Simple RAG Demo")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()

question = st.text_input("Ask a question about the PDF")

if question:

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    st.subheader("Answer")

    st.write(response.content)

    with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"### Chunk {i}")
            st.write(doc.page_content)