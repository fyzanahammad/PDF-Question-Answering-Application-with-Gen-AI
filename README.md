
# PDF Question Answering Application with Gen AI

This repository contains a Streamlit application that enables users to upload multiple PDF documents and ask questions based on the content of these documents. The application utilizes Google Gemini-pro and Embedding model to process text and generate answers, providing an interactive and intuitive interface for users to interact with AI in a meaningful way.

## Features
- **PDF Upload**: Users can upload multiple PDF files from which the application extracts text.
- **Question Answering**: Users can input questions, and the application provides answers based on the content of the uploaded PDFs.

## How It Works
![WORKflow](https://github.com/fyzanahammad/PDF-Question-Answering-Application-with-Gen-AI/assets/72655841/3778fdb6-61e3-4e48-a36e-fb4f9ed320bc)

1. The application processes the uploaded PDF files to extract text,
2. Which is then split into manageable chunks.
3. These chunks are converted into embeddings using Google Generative AI embeddings, and a FAISS index is created for efficient similarity search.
4. When a question is asked, the application searches for the most relevant text chunks and uses a conversational model to generate an answer.

## Technologies Used
- **Streamlit**: Streamlit is an open-source framework that turns data scripts into web apps in minutes. You can use Streamlit to create interactive and powerful data apps with Python, no front-end experience required, and deploy them on Community Cloud for free.
- **PyPDF2**: A Pure-Python library built as a PDF toolkit. It is capable of extracting document information, splitting documents page by page, merging documents, cropping pages, and more.
- **LangChain and LangChain Google GenAI**: Libraries used for chaining language models together to create applications. These libraries provide tools for handling text data, creating conversational AI chains, and integrating with Google's Generative AI models for embedding and chat functionalities.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors. It is used in this project to store and search embeddings derived from the PDF texts.
- **Dotenv**: A zero-dependency module that loads environment variables from a `.env` file into `os.environ`.
- **Google Generative AI**: Provides access to powerful models for embeddings and conversational AI, which are pivotal in processing and answering user queries based on the document content.

## Setup and Installation
1. Clone the repository:
   ```
   git clone https://github.com/fyzanahammad/PDF-Question-Answering-Application-with-Gen-AI
   ```
2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up a `.env` file in the root directory with your Google API key:
   ```
   GOOGLE_API_KEY='your_api_key_here'
   ```
4. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

## Contributing
Feel free to reach out, if you find bugs or have suggestions for improvements.
