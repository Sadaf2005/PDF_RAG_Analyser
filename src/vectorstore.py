import os
import shutil
from typing import List, Any
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.utils import load_processed_data, split_text_into_chunks
from src.embeddings import get_embedding_function


# Configuration
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "rag_capstone_collection_v1"


def create_vector_store(chunks: List[Document], embedding_function: Any):
    """
    Truncates and rebuilds a Chroma vector store from a list of LangChain Documents.
    """
    print(f"\nStarting ChromaDB update at: {CHROMA_DB_PATH}")

    vectordb = Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function
    )

    try:
        existing = vectordb.get()
        ids = existing.get("ids", [])
        print(f"Found {len(ids)} existing documents.")
        if ids:
            vectordb.delete(ids=ids)
            vectordb.persist()
            print("Existing documents deleted.")
        else:
            print("No documents to delete.")
    except Exception as e:
        print(f"Error while deleting documents: {e}")

    print(f"Adding {len(chunks)} new chunks...")
    vectordb.add_documents(chunks)
    vectordb.persist()
    print("ChromaDB updated successfully.")

    return vectordb
    
    

    
    


def run_simple_similarity_query(vectordb: Chroma, query: str = "What are the core components of RAG?") -> None:
    """
    Runs a simple similarity search against the new vector store.
    """
    print("\n--- Running Simple Similarity Query (Checkpoint E Test) ---")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)
    
    print(f"Query: '{query}'")
    print(f"Retrieved {len(results)} relevant documents.")
    
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source')} | Title: {doc.metadata.get('title')}")
        print(f"Content Snippet: {doc.page_content[:200]}...")


if __name__ == "__main__":
    
    try:
        loaded_docs = load_processed_data()
        chunks = split_text_into_chunks(loaded_docs)
        
        if not chunks:
            print("\nFATAL ERROR: No chunks were created. Cannot proceed. Check data directories and utils.py.")
        else:
            embedding_func = get_embedding_function()
            
            vector_db = create_vector_store(chunks, embedding_func)
            
            run_simple_similarity_query(vector_db)
            
            print("\n(Embeddings and ChromaDB) complete")

    except ImportError as e:
        print(f"\nERROR: Missing dependencies. Please ensure `src/utils.py` and `src/embeddings.py` exist and you have installed all requirements: {e}")