https://financial-pdf-analyser-45.streamlit.app/

📈 Financial PDF Analyzer with Gemini 1.5
An intelligent Streamlit web application that allows you to interact with multiple PDF files using Google's Gemini 1.5 AI model. This tool is specifically designed to extract insights, analyze financial data, and answer questions based on the content of uploaded documents like annual reports, financial statements, and research papers.

✨ Key Features
📄 Multi-PDF Upload: Upload one or more PDF documents simultaneously.

🤖 AI-Powered Q&A: Ask complex questions in natural language and receive context-aware answers.

🧠 Advanced AI Model: Powered by LangChain and Google's powerful gemini-1.5-flash model for high-quality, contextual responses.

🗃️ Efficient Data Processing: Uses GoogleGenerativeAIEmbeddings and a FAISS vector database for rapid and relevant information retrieval.

📊 Specialized for Finance: Optimized prompts for analyzing financial reports, related-party transactions, executive remuneration, and business performance.

🗨️ Interactive Chat Interface: A user-friendly, chat-like interface with user/bot avatars to track your conversation.

📥 Export Conversation: Download your entire chat history as a CSV file for your records.

🏛️ How It Works: RAG Architecture
This application is built on a Retrieval-Augmented Generation (RAG) architecture.

PDF Processing: The text from your uploaded PDFs is extracted and split into smaller, manageable chunks.

Embedding: Each text chunk is converted into a numerical vector representation (an "embedding") using Google's embedding model. These embeddings capture the semantic meaning of the text.

Vector Store: The embeddings are stored in a FAISS vector database, which allows for incredibly fast similarity searches.

Retrieval & Generation:

When you ask a question, your query is also converted into an embedding.

The FAISS database finds the text chunks from the original PDFs that are most relevant to your question.

These relevant chunks, along with your original question, are sent to the Gemini 1.5 model as a detailed prompt.

The model generates a precise answer based only on the context provided, ensuring factual accuracy and preventing hallucinations.

🛠️ Tech Stack
Framework: Streamlit

LLM Orchestration: LangChain

Large Language Model: Google Gemini 1.5 Flash

Embeddings: Google Generative AI Embeddings

PDF Processing: PyPDF2

Vector Database: FAISS (Facebook AI Similarity Search)

Data Handling: Pandas

Deployment: Streamlit Community Cloud

🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.

Prerequisites
Python 3.9 or higher

A Google AI API Key. You can get one from Google AI Studio.

Installation & Setup
Clone the repository:

git clone [https://github.com/your-username/financial-pdf-analyzer.git](https://github.com/your-username/financial-pdf-analyzer.git)
cd financial-pdf-analyzer

Create and activate a virtual environment:

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install the required dependencies:

pip install -r requirements.txt

Set up your environment variables:

Create a file named .env in the root of the project directory.

Add your Google API key to this file:

GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"

Running the Application
Launch the Streamlit app with the following command:

streamlit run app.py

The application will open in a new tab in your web browser.

📖 How to Use
Upload PDFs: Use the sidebar to upload the financial reports you want to analyze.

Process Documents: Click the "Process Documents" button. The app will extract text, create embeddings, and build the vector store. This may take a moment.

Ask Questions: Once processing is complete, type your questions into the chat input at the bottom of the page.

Export History: If you wish to save your conversation, click the "Download Chat as CSV" button in the sidebar.
