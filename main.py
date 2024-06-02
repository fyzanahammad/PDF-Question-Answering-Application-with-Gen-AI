import os
import subprocess
import asyncio
import streamlit as st
import google.generativeai as genai
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

def convert_pdf_to_markdown(input_path, output_folder, langs='English'):
    output_dir = os.path.join(output_folder, os.path.basename(input_path).replace('.pdf', ''))
    command = [
        "marker_single",
        input_path,
        output_folder,
        "--langs", langs
    ]
    subprocess.run(command, check=True)
    output_file = os.path.join(output_dir, os.path.basename(input_path).replace('.pdf', '.md'))
    return output_file

def get_markdown_text(markdown_file):
    with open(markdown_file, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def get_unique_filename(base_path, base_name, extension):
    counter = 0
    new_path = f"{base_path}_{counter}{extension}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}_{counter}{extension}"
    return new_path

def get_pdf_text(pdf_docs, output_folder, langs='English', progress=None):
    text = ""
    total_files = len(pdf_docs)
    for i, pdf in enumerate(pdf_docs):
        if progress:
            progress.progress((i + 1) / total_files)
            st.session_state.status_text.text(f"Processing {pdf.name} ({i + 1}/{total_files})")
        
        # Generate a new file name to avoid any issues with original file names
        base_path = os.path.join(output_folder, f"temp_pdf_{i}")
        pdf_path = get_unique_filename(base_path, f"temp_pdf_{i}", ".pdf")
        with open(pdf_path, 'wb') as f:
            f.write(pdf.getbuffer())
        
        markdown_file = convert_pdf_to_markdown(pdf_path, output_folder, langs)
        text += get_markdown_text(markdown_file) + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

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

# Language selection dropdown
langs = st.selectbox("Select language", options=['English', 'Spanish', 'French', 'German', 'Chinese', 'Japanese'])

# File uploader allows user to upload multiple PDFs
uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)
if uploaded_files and 'pdf_texts' not in st.session_state:
    # Temporary directory to store markdown files
    output_folder = 'temp_output'
    os.makedirs(output_folder, exist_ok=True)

    # Progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()
    st.session_state.status_text = status_text
    
    # Extract text from PDFs
    pdf_texts = get_pdf_text(uploaded_files, output_folder, langs, progress=progress_bar)
    
    # Update status
    status_text.text("Splitting text into chunks...")
    text_chunks = get_text_chunks(pdf_texts)
    
    # Update status
    status_text.text("Creating vector store...")
    vector_store = get_vector_store(text_chunks)
    
    # Save processed data to session state
    st.session_state.pdf_texts = pdf_texts
    st.session_state.text_chunks = text_chunks
    st.session_state.vector_store = vector_store
    
    # Update status
    status_text.text("PDFs processed and vector store updated!")
    progress_bar.progress(1.0)
    st.success('PDFs processed and vector store updated!')

# User question input
user_question = st.text_input("Enter your question here:")
if st.button("Ask") and user_question and 'vector_store' in st.session_state:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = st.session_state.vector_store
    docs = new_db.similarity_search(user_question)

    # Create an event loop for the asynchronous chain
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    chain = loop.run_until_complete(get_conversational_chain())
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Display response
    st.text("Q: " + user_question)
    st.write("A: " + response["output_text"])
