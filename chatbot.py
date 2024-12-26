import sys
import os


# Get the directory containing langchain
path2 = "/Users/jardenna/opt/anaconda3/envs/auto5/lib/python3.11/site-packages"

# Add this directory to sys.path
sys.path.append(path2)

import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
# from langchain.embeddings import Embedding

from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import ollama
from langchain_community.chat_models import ChatOllama

# import PyPDF2
import io
import requests

import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()


# Streamlit UI Header
st.title("Document Chatbot")

# Sidebar for file upload
st.sidebar.header("Upload a PDF Document")


# # Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file:
    # Extract text from PDF
    with st.spinner("Extracting text from the document..."):
        document_text = extract_text_from_pdf(uploaded_file)

    # Display extracted text
    st.subheader("Extracted Text")
    st.text_area("Document Content", document_text, height=300)


    # Debugging: Check the length of the extracted text
    text_length = len(document_text)
    st.write(f"Extracted text length: {text_length} characters.")

    # Chunking the text
    with st.spinner("Chunking the text into paragraphs..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(document_text)


    # Path to Chroma storage
    persist_directory = "/app/chroma_storage"
    
    # Remove previous vector store if it exists, so it gets overwritten
    if os.path.exists(persist_directory):
        st.warning(f"Previous vector store found. Deleting existing vector store...")
        for filename in os.listdir(persist_directory):
            file_path = os.path.join(persist_directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Initialize the embeddings model (Ollama LLM)
    with st.spinner("Embedding text using Ollama LLM..."):

        url = "http://ollama:11434/api/embed"


        # Initialize the OllamaEmbeddings class
        embedding_function = OllamaEmbeddings(model="llama3.2", base_url="http://ollama:11434")


        # Initialize a list to store the embeddings
        embedding_vectors = []

        # Loop through the chunks of text and get embeddings using OllamaEmbeddings
        for chunk in chunks:
            try:
                # Get the embedding for the current chunk of text using embedding_function
                embedding_vector = embedding_function.embed_query(chunk)  # This method directly returns the embedding vector
                if embedding_vector:
                    embedding_vectors.append(embedding_vector)  # Collect the embedding vector
                else:
                    print(f"No embedding found for chunk: {chunk}")
            except Exception as e:
                print(f"Error embedding chunk: {e}")

        # Initialize the Chroma vector store with the embedding function
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)


        # Add the chunks and embeddings to Chroma
        with st.spinner("Adding embeddings to vector store..."):
            for i, (chunk, embedding_vector) in enumerate(zip(chunks, embedding_vectors)):
                # Add the chunk and its corresponding embedding vector to Chroma
                vectorstore.add_texts([chunk], metadatas=[{"chunk_index": i}], embeddings=[embedding_vector])

            # Persist the vector store to disk
            vectorstore.persist()


    st.success("Text successfully embedded and stored in the vector database!")

    # Chat functionality
    st.subheader("Chat with the Document")
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        with st.spinner("Retrieving the most relevant context..."):
            # Retrieve the most relevant chunks
            results = vectorstore.similarity_search(query=user_question, k=5)

            # Combine context for response generation
            if results:
                try:
                    context = "\n".join([getattr(result, "page_content", "") for result in results])
                except AttributeError:
                    context = "Document objects do not have a page_content attribute."
            else:
                context = "No results found."



        with st.spinner("Generating an answer..."):
            # Initialize the Ollama model (make sure the base_url is correct)
            ollama = ChatOllama(model="llama3.2", base_url="http://ollama:11434")

            # Define the messages to send to the model
            messages = [
                ("system", "You are an assistant that answers questions based on the provided document context."),
                ("assistant", f"Context: {context}"),
                ("human", user_question)  # The user's input question
            ]

            # Generate the response from Ollama using invoke()
            response = ollama.invoke(messages)

            # Extract and display the response text
            response_text = response.content  # The response text will be in the 'text' attribute
            st.markdown(f"**Answer:** {response_text}")