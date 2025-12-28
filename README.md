# 🧠 RAG Research Assistant with Groq & LangChain

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-green)
![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple)

> **Capstone Project** > **Author:** Shubham Mukherjee  
> **Department:** Computer Science and Engineering, Nitte Meenakshi Institute of Technology

## 📖 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system designed to enhance the factual accuracy and reasoning capabilities of Large Language Models (LLMs). [cite_start]By integrating **Groq's high-speed inference engine**, **LangChain**, and **ChromaDB**, this tool allows users to build a custom knowledge base from PDF research papers and live web pages[cite: 17, 18].

[cite_start]Unlike standard LLMs that rely solely on pre-trained data, this system retrieves contextually relevant information from your uploaded documents to provide accurate, domain-specific answers with source citations[cite: 21, 61].

## 🚀 Key Features
* [cite_start]**Multi-Source Ingestion:** Custom tools to crawl web pages (`BeautifulSoup`) and scrape PDF research papers (`pdfplumber`)[cite: 19].
* [cite_start]**High-Speed Inference:** Powered by **Groq** (using `llama-3.1-8b-instant`) for near real-time responses[cite: 96].
* [cite_start]**Semantic Search:** Uses **ChromaDB** to store and retrieve document embeddings based on meaning rather than just keywords[cite: 20].
* [cite_start]**Smart Embeddings:** Text is chunked and embedded using HuggingFace’s `all-MiniLM-L6-v2` model[cite: 147].
* [cite_start]**Hallucination Reduction:** Answers are grounded in retrieved data, significantly reducing AI hallucinations[cite: 22].

## 🛠️ Tech Stack
* **Language:** Python
* **Orchestration:** LangChain
* **LLM Provider:** Groq (Llama 3.1)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace / Sentence Transformers
* **Data Extraction:** BeautifulSoup4 (Web), pdfplumber (PDF)
* **Frontend:** Streamlit

## ⚙️ Architecture
The system follows a standard RAG pipeline:
1.  **Ingest:** Documents (PDFs/URLs) are scraped and cleaned.
2.  [cite_start]**Chunk:** Text is split into segments (~512 chars) with overlap[cite: 143].
3.  **Embed:** Chunks are converted to vectors using `all-MiniLM-L6-v2`.
4.  [cite_start]**Store:** Vectors are saved locally in ChromaDB[cite: 151].
5.  **Retrieve:** User queries trigger a semantic search for the top-k most relevant chunks.
6.  [cite_start]**Generate:** The Groq LLM answers the question using the retrieved context[cite: 187].

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/rag-research-assistant.git](https://github.com/YOUR_USERNAME/rag-research-assistant.git)
    cd rag-research-assistant
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Core libraries used:* `langchain`, `langchain_groq`, `chromadb`, `beautifulsoup4`, `requests`, `pdfplumber`, `python-dotenv`[cite: 87].

4.  **Configure Environment Variables**
    Create a `.env` file in the root directory and add your Groq API key:
    ```env
    GROQ_API_KEY="gsk_your_api_key_here"
    GROQ_MODEL="llama-3.1-8b-instant"
    ```
    *You can get a free API key from the [Groq Console](https://console.groq.com/).*

## 🖥️ Usage

**1. Run the Application**
Launch the Streamlit interface:
```bash
streamlit run streamlit_app.py
