import os
import json
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.documents import Document

# --- Configuration ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
DATA_DIRS = ["data/raw_web", "data/papers/processed"]

def load_processed_data(data_directories: List[str] = DATA_DIRS) -> List[Dict[str, Any]]:
    """
    Loads all processed JSON files from the specified data directories.

    Args:
        data_directories: A list of directories to search for processed data.

    Returns:
        A list of dictionaries, where each dictionary represents a loaded document.
    """
    all_documents = []
    print(f"Loading data from: {data_directories}...")
    
    for directory in data_directories:
        for filename in os.listdir(directory):
            if filename.endswith("_processed.json"):
                file_path = os.path.join(directory, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        if data.get('text') or data.get('full_text'):
                            all_documents.append(data)
                        else:
                            print(f"Skipping {filename}: No text content found.")
                            
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {filename}. Skipping.")
                except Exception as e:
                    print(f"An unexpected error occurred while loading {filename}: {e}. Skipping.")
                    
    print(f"Successfully loaded {len(all_documents)} documents.")
    return all_documents


def get_text_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> TextSplitter:
    """
    Initializes and returns the RecursiveCharacterTextSplitter.
    """
    separators = [
        "\n\n",  
        "\n",    
        " ",     
        "",
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len, 
        is_separator_regex=False
    )
    return splitter


def split_text_into_chunks(documents: List[Dict[str, Any]]) -> List[Document]:
    """
    Splits the content of loaded documents into manageable chunks (LangChain Documents).
    """
    splitter = get_text_splitter()
    
    chunks = []
    
    for doc in documents:
        content_text = doc.get('full_text') or doc.get('text')
        
        if not content_text:
            continue
            
        metadata = {k: v for k, v in doc.items() if k not in ['text', 'full_text']}

        source_document = Document(page_content=content_text, metadata=metadata)
        
        doc_chunks = splitter.split_documents([source_document])
        chunks.extend(doc_chunks)

    print(f"Original documents split into a total of {len(chunks)} chunks.")
    return chunks


# --- Test Function ---
if __name__ == "__main__":
    #Dummy Data
    web_data = {
        "url": "https://example.com/web",
        "title": "Web Article",
        "source": "web_page",
        "text": "This is a web article. " * 300 
    }
    web_file_path = os.path.join("data/raw_web", "web_article_processed.json")
    if not os.path.exists(web_file_path):
        os.makedirs("data/raw_web", exist_ok=True)
        with open(web_file_path, 'w') as f:
            json.dump(web_data, f)
            
    pdf_data = {
        "path": "data/papers/example.pdf",
        "title": "Academic Paper Title",
        "source": "pdf_paper",
        "abstract": "The abstract summary.",
        "full_text": "The full body of the academic paper, including introduction, methods, and results. " * 500
    }
    pdf_file_path = os.path.join("data/papers/processed", "pdf_paper_processed.json")
    if not os.path.exists(pdf_file_path):
        os.makedirs("data/papers/processed", exist_ok=True)
        with open(pdf_file_path, 'w') as f:
            json.dump(pdf_data, f)

    print("\n--- Running Chunking Utility Test ---")
    
    loaded_docs = load_processed_data()
    
    final_chunks = split_text_into_chunks(loaded_docs)
    
    print("\n--- Chunking Results Summary ---")
    print(f"Total Chunks Created: {len(final_chunks)}")
    
    if final_chunks:
        print("\nExample Chunk 1:")
        print(f"  Content (first 200 chars): {final_chunks[0].page_content[:200]}...")
        print(f"  Metadata: {final_chunks[0].metadata}")
        
        if len(final_chunks) > 1:
            print("\nExample Chunk 2:")
            print(f"  Content (first 200 chars): {final_chunks[1].page_content[:200]}...")
            print(f"  Metadata: {final_chunks[1].metadata}")
        
    print("\nChunking utility test complete.")