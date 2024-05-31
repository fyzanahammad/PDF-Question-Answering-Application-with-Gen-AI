import os
import streamlit as st
import google.generativeai as genai
import asyncio

from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure GenAI with the API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

async def get_conversational_chain():
    template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not available in the context, please state so.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = load_qa_chain(model, prompt=prompt, verbose=True)
    return chain

# Streamlit interface
st.title('PDF Question Answering with Gen AI')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# File uploader allows user to upload multiple PDFs
uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)
if uploaded_files:
    pdf_texts = get_pdf_text(uploaded_files)
    text_chunks = get_text_chunks(pdf_texts)
    get_vector_store(text_chunks)
    st.success('PDFs processed and vector store updated!')

# User question input
user_question = st.text_input("Enter your question here:")
if user_question:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = asyncio.run(get_conversational_chain())
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Update chat history
    st.session_state.chat_history.append("Q: " + user_question)
    st.session_state.chat_history.append("A: " + response["output_text"])
    
    # Display chat history
    for message in st.session_state.chat_history:
        st.text(message)
