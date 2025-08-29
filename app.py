import streamlit as st
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# ADD this new line:
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


# ADD THESE THREE LINES
import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
# END OF ADDED LINES
# Load environment variables from .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Function to extract text from a list of PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create a FAISS vector store from text chunks
def get_vector_store(text_chunks):
    if not text_chunks:
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Persist the vector store in session state
    st.session_state.vector_store = vector_store


# Function to create the conversational chain
def get_conversational_chain():
    prompt_template = """
    You are a specialized financial analyst AI. Your task is to provide detailed and accurate answers based on the provided context from annual reports and financial documents.

    Context:
    {context}

    Question:
    {question}

    Answer:
    Based on the documents, """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate a response
def handle_user_input(user_question):
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.warning("Please upload and process your PDF documents first.")
        return

    # Retrieve relevant documents from the vector store
    docs = st.session_state.vector_store.similarity_search(user_question)
    
    # Get the conversational chain and run it
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Add the conversation to chat history
    st.session_state.chat_history.append(("You", user_question))
    st.session_state.chat_history.append(("Bot", response["output_text"]))



# --- STREAMLIT UI ---

# Page configuration
st.set_page_config(page_title="Financial Report Analyzer 📈", layout="wide")

# New and updated CSS block
st.markdown("""
<style>
    .chat-container { display: flex; flex-direction: column; width: 100%; margin: auto; }
    .chat-bubble { 
        padding: 10px 15px; 
        border-radius: 15px; 
        margin-bottom: 10px; 
        max-width: 70%; 
        word-wrap: break-word;
        color: black; /* This line makes the text black */
    }
    .user-bubble { background-color: #DCF8C6; align-self: flex-end; border-bottom-right-radius: 0; }
    .bot-bubble { background-color: #F1F0F0; align-self: flex-start; border-bottom-left-radius: 0; }
    .avatar { font-size: 20px; margin-right: 10px; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history and vector store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar ---
with st.sidebar:
    st.title("📄 Financial PDF Analyzer")
    st.write("Upload annual reports to ask questions and extract insights.")
    
    if not google_api_key:
        st.error("Google API Key not found. Please set it in your .env file.")
    else:
        st.success("API Key loaded successfully.")

    st.header("Upload Documents")
    pdf_docs = st.file_uploader("Upload PDF files here", accept_multiple_files=True, type="pdf")
    
    if st.button("Process Documents"):
        if pdf_docs:
            with st.spinner("Processing documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            st.success("Done! You can now ask questions.")
        else:
            st.warning("Please upload at least one PDF file.")

    if st.session_state.chat_history:
        st.header("Export Conversation")
        df = pd.DataFrame(st.session_state.chat_history, columns=["Speaker", "Message"])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Chat as CSV", csv, 'chat_history.csv', 'text/csv')

# --- Main Chat Interface ---
st.header("Chat with your Documents 🤖")

chat_container = st.container()
with chat_container:
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f'<div class="chat-bubble user-bubble"><span class="avatar">👤</span>{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble bot-bubble"><span class="avatar">🤖</span>{message}</div>', unsafe_allow_html=True)

user_question = st.chat_input("Ask a question about your documents...")
if user_question:
    handle_user_input(user_question)
    st.rerun()