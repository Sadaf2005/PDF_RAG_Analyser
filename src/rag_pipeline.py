import os
from typing import Dict, List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

from src.embeddings import get_embedding_function
from src.vectorstore import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME 


load_dotenv()


GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
K_RETRIEVAL = 5 

RAG_PROMPT_TEMPLATE = """You are an expert Q&A assistant. Your goal is to answer the user's question 
based ONLY on the provided context, which consists of research paper and web snippets.
If the context does not contain the answer, state clearly that you cannot find the answer in the provided documents.
For every fact you state, cite the 'title' and 'source' from the metadata of the document(s) you used.

CONTEXT:
{context}

QUESTION:
{question}
"""

def build_rag_chain() -> RetrievalQA:
    """
    Initializes the RAG components (LLM, Embeddings, Vector Store) and builds the RetrievalQA chain.
    """
    print("--- Initializing RAG Pipeline Components ---")

   
   
    embedding_func = get_embedding_function()

   
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(f"ChromaDB not found at {CHROMA_DB_PATH}. Please run src/vectorstore.py first.")
        
    vectordb = Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_func
    )
    print("ChromaDB loaded successfully.")

    retriever = vectordb.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": K_RETRIEVAL}
    )
    print(f"Retriever configured to fetch top {K_RETRIEVAL} chunks.")

    llm = ChatGroq(model_name=GROQ_MODEL, temperature=0) #factual
    print(f"Groq LLM initialized with model: {GROQ_MODEL}")

    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)}
    )

    print("RetrievalQA Chain built successfully.")
    return qa_chain

def run_rag_query(qa_chain: RetrievalQA, query: str):
    """
    Executes a query against the RAG chain and prints the results.
    """
    print(f"\n--- Running Query: '{query}' ---")
    
    response = qa_chain.invoke({"query": query})

    print("\n[--- FINAL LLM ANSWER ---]")
    print(response['result'])
    print("[-------------------------]")
    
    print("\n[--- RETRIEVED SOURCES ---]")
    sources = set()
    for doc in response['source_documents']:
        sources.add(f"Title: {doc.metadata.get('title', 'N/A')} | Source: {doc.metadata.get('source', 'N/A')} | URL/Path: {doc.metadata.get('url', doc.metadata.get('path', 'N/A'))}")
    
    for source in sources:
        print(f" - {source}")
    print("[-------------------------]")
    
    return response



def run_evaluation_suite(qa_chain: RetrievalQA, test_set: List[Dict[str, str]]):
    """
    Runs a batch of queries for manual evaluation and logs results.
    """
    print("\n" + "="*50)
    print("      RUNNING EVALUATION SUITE")
    print("="*50)
    
    evaluation_results = []
    
    for i, test_case in enumerate(test_set):
        print(f"\n--- TEST CASE {i+1}/{len(test_set)} ---")
        query = test_case["query"]
        
       
        response = run_rag_query(qa_chain, query)
        
        
        sources = [
            doc.metadata.get('title', 'N/A')
            for doc in response.get('source_documents', [])
        ]
        
        result_entry = {
            "ID": i + 1,
            "Query": query,
            "Expected_Fact": test_case["expected_fact"],
            "LLM_Response_Snippet": response['result'][:150].replace('\n', ' ') + '...',
            "Retrieved_Sources": ", ".join(set(sources)),
            "Relevance_Score": "TBD (Manual)",
            "Correctness_Score": "TBD (Manual)"
        }
        evaluation_results.append(result_entry)
        
    print("\n" + "="*50)
    print("EVALUATION RESULTS TABLE (For Manual Scoring)")
    print("="*50)
    
    
    headers = ["ID", "Query", "Expected_Fact", "LLM_Response_Snippet", "Retrieved_Sources"]
    
    
    for entry in evaluation_results:
        print(f"ID: {entry['ID']}")
        print(f"  Q: {entry['Query']}")
        print(f"  E: {entry['Expected_Fact']}")
        print(f"  A: {entry['LLM_Response_Snippet']}")
        print(f"  S: {entry['Retrieved_Sources']}")
        print("-" * 20)
        




TEST_QUERIES = [
    {
        "query": "What are the core components of Retrieval-Augmented Generation?",
        "expected_fact": "RAG typically involves a retriever (e.g., vector database) and a generator (LLM)."
    },
    {
        "query": "Can you summarize the web article about the Llama 3.1 8B model?",
        "expected_fact": "Should mention key features or speeds of the Groq-hosted Llama 3.1 8B model."
    },
    {
        "query": "According to the PDF papers, what is one major limitation of standard transformer models?",
        "expected_fact": "Should reference a paper's conclusion/intro discussing attention or context size limits."
    }
]

if __name__ == "__main__":
    try:
       
        rag_chain = build_rag_chain()
        
       
        run_evaluation_suite(rag_chain, TEST_QUERIES)

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please ensure you run the data scraping and vectorstore creation steps first (python -m src.vectorstore).")
    except Exception as e:
        print(f"\nAn error occurred during RAG pipeline execution: {e}")