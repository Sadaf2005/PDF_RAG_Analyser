from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_core.embeddings import Embeddings


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_function() -> Embeddings:
    """
    Initializes and returns the chosen embedding function.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    
    
    # useing 'SentenceTransformers' via LangChain's wrapper
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, 
        model_kwargs={'device': 'cpu'} 
    )
    
    print("Embedding model loaded successfully.")
    return embedding_function

if __name__ == "__main__":
    emb_func = get_embedding_function()
    test_vector = emb_func.embed_query("What is RAG?")
    print(f"\nTest embedding generated. Vector dimension: {len(test_vector)}")