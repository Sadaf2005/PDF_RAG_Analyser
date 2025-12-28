from langchain_community.vectorstores import Chroma
from src.embeddings import get_embedding_function
from src.vectorstore import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME

# Loading embedding function
embedding_func = get_embedding_function()

# Loading ChromaDB
vectordb = Chroma(
    persist_directory=CHROMA_DB_PATH,
    collection_name=CHROMA_COLLECTION_NAME,
    embedding_function=embedding_func
)


docs = vectordb.get()
print(f"Total documents: {len(docs['ids'])}")


for i in range(min(5, len(docs['ids']))):
    print(f"\nDocument ID: {docs['ids'][i]}")
    print(f"Metadata: {docs['metadatas'][i]}")
    print(f"Content snippet: {docs['documents'][i][:200]}...")






