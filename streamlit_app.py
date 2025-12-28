import streamlit as st
import os
import sys
from dotenv import load_dotenv
from src.tools.pdf_scraper import extract_pdf_sections
from src.tools.web_crawler import fetch_page_text
from src.utils import load_processed_data, split_text_into_chunks
from src.embeddings import get_embedding_function
from src.vectorstore import create_vector_store


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


try:
    from src.rag_pipeline import build_rag_chain 
    from src.vectorstore import CHROMA_DB_PATH
except ImportError as e:
    st.error(f"Failed to import core RAG components. Please ensure all files are in the 'src/' directory and requirements are installed. Error details: {e}")
    st.stop()

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# --- Streamlit Caching ---
@st.cache_resource
def load_rag_chain():
    """
    Initializes and returns the RAG chain built in src/rag_pipeline.py.
    This function is cached to prevent expensive re-initialization on every user interaction.
    """
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set in your environment or .env file.")
        return None
    
    if not os.path.exists(CHROMA_DB_PATH):
        st.warning(
            "ChromaDB not found. Please run 'python -m src.vectorstore' first "
            "to create the knowledge base from your processed data."
        )
        return None

    with st.spinner("Initializing RAG components (LLM, Embeddings, Vector Store)..."):
        try:
            
            return build_rag_chain()
        except Exception as e:
            st.error(f"Error during RAG chain initialization: {e}")
            return None

# --- Main Streamlit App Layout ---

st.set_page_config(page_title="Groq-Powered RAG Capstone", layout="wide")
st.title("⚡ Groq RAG Research Assistant")
st.markdown("Retrieval-Augmented Generation using LangChain, ChromaDB, and Groq for blazing-fast inference.")

qa_chain = load_rag_chain()

with st.sidebar:
    st.header("Setup & Configuration")
    if qa_chain and hasattr(qa_chain, 'llm'):
        st.markdown(f"**LLM Model:** `{qa_chain.llm.model_name}`")
    else:
        st.markdown("**LLM Model:** Loaded")
        
    st.markdown(f"**Vector Store:** ChromaDB (Path: `{CHROMA_DB_PATH}`)")
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown(
        "1. **Scrape Data:** Run your scrapers (`src/tools/*`).\n"
        "2. **Build DB:** Run `python -m src.vectorstore`.\n"
        "3. **Query:** Enter a research question in the chat box below."
    )
    if st.button("Clear Cache & Re-Initialize"):
        st.cache_resource.clear()
        st.rerun()




st.markdown("## 📥 Upload PDFs or Enter URLs")

with st.form("upload_form"):
    uploaded_pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    url_input = st.text_area("Paste one or more URLs (one per line)")
    submitted = st.form_submit_button("Ingest Data")

import glob

if submitted:
    with st.spinner("Processing uploaded data..."):
        
        st.cache_resource.clear()

       
        pdf_dir = "data/papers"
        if os.path.exists(pdf_dir):
            for f in glob.glob(os.path.join(pdf_dir, "*.pdf")):
                os.remove(f)
        else:
            os.makedirs(pdf_dir)

        
        web_dir = "data/raw_web"
        if os.path.exists(web_dir):
            for f in glob.glob(os.path.join(web_dir, "*")):
                os.remove(f)
        else:
            os.makedirs(web_dir)

        
        if uploaded_pdfs:
            for pdf in uploaded_pdfs:
                pdf_path = os.path.join(pdf_dir, pdf.name)
                with open(pdf_path, "wb") as f:
                    f.write(pdf.read())
                extract_pdf_sections(pdf_path)

        if url_input.strip():
            for url in url_input.strip().splitlines():
                if url.strip():
                    fetch_page_text(url.strip())

        
        docs = load_processed_data()
        chunks = split_text_into_chunks(docs)
        embedding_func = get_embedding_function()
        create_vector_store(chunks, embedding_func)

        st.success("Previous data cleared. New data ingested and vector store updated successfully!")
        st.rerun()






# --- Chat Interface ---

if qa_chain is None:
    
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question about your research papers or web articles..."):
   
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

   
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Searching and generating response... (Groq is fast!)"):
            try:
               
                response = qa_chain.invoke({"query": prompt})

                llm_answer = response['result']
                source_documents = response['source_documents']

                
                sources_text = "### 📚 Sources Used\n\n"
                
                unique_sources = set()
                for doc in source_documents:
                    title = doc.metadata.get('title', 'Untitled Document')
                    source = doc.metadata.get('source', 'Unknown Source')
               
                    url_or_path = doc.metadata.get('url', doc.metadata.get('path', 'N/A'))
                    unique_sources.add((title, source, url_or_path))

                for title, source, url_or_path in unique_sources:
                    sources_text += f"- **{title}** ({source}): `{url_or_path}`\n"

                full_response_markdown = f"**{llm_answer}**\n\n---\n\n{sources_text}"
                
                message_placeholder.markdown(full_response_markdown)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response_markdown})
                
            except Exception as e:
                error_message = f"An error occurred during RAG query: {e}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
