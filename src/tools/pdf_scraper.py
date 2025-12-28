import pdfplumber
import re
import os
import json
from typing import Dict, Any

# --- Configuration ---
PROCESSED_DATA_DIR = "data/papers/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- Heuristic Regex Patterns ---
ABSTRACT_REGEX = re.compile(
    r"abstract\s*(.*?)(?:^\s*1\s*introduction|^introduction|^\s*[A-Z][a-z]+[A-Z][a-z]+|^keywords|^\s*i\s*introduction)", 
    flags=re.I | re.M | re.S
)

INTRO_REGEX = re.compile(
    r"(?:^\s*1\s*introduction|^introduction)(.*?)(?:^\s*[2-9]\s*[A-Z\s]+|^\s*[a-z]\s*\.)", 
    flags=re.I | re.M | re.S
)

CONCLUSION_REGEX = re.compile(
    r"(?:^conclusion|^conclusions)(.*?)(?:^\s*acknowledgements|^references|^\s*appendix)", 
    flags=re.I | re.M | re.S
)


# --- Core Function ---
def extract_pdf_sections(path: str) -> Dict[str, Any]:
    """
    Extracts text and key sections from a local PDF file.

    Args:
        path: Local file path to the PDF.

    Returns:
        A dictionary with extracted content and metadata.
    """
    print(f"Processing PDF: {path}")
    
    metadata = {
        "path": path,
        "title": os.path.basename(path),
        "abstract": "",
        "introduction": "",
        "conclusion": "",
        "full_text": "",
        "source": "pdf_paper"
    }

    if not os.path.exists(path):
        metadata["error"] = "File not found."
        print(metadata["error"])
        return metadata

    try:
        with pdfplumber.open(path) as pdf:
            full_text_list = [page.extract_text() or "" for page in pdf.pages]
            full_text = "\n".join(full_text_list)
            
            normalized_text = full_text.lower()
            
            first_page_text = full_text_list[0]
            metadata["title"] = first_page_text.split('\n')[0].strip()

            
            def safe_search(regex: re.Pattern, text: str, default: str = "") -> str:
                """Helper function to run regex search and return cleaned group 1."""
                match = regex.search(text)
                if match:
                    return " ".join(match.group(1).strip().split())
                return default

            metadata["abstract"] = safe_search(ABSTRACT_REGEX, normalized_text)
            metadata["introduction"] = safe_search(INTRO_REGEX, normalized_text)
            metadata["conclusion"] = safe_search(CONCLUSION_REGEX, normalized_text)

            if not metadata["abstract"]:
                metadata["abstract"] = " ".join(full_text.split()[:150])
                
            metadata["full_text"] = " ".join(full_text.split())
            
            doc_id = os.path.basename(path).replace(".pdf", "").replace(" ", "_").lower()
            output_path = os.path.join(PROCESSED_DATA_DIR, f"{doc_id}_processed.json")
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in metadata.items() if k != "full_text"}, f, indent=2) 
            
            print(f"Successfully extracted. Abstract length: {len(metadata['abstract'])}. Saved metadata to: {output_path}")
            
            return metadata

    except Exception as e:
        metadata["error"] = f"Error processing PDF: {e}"
        print(metadata["error"])
        return metadata

# --- Test function ---
if __name__ == "__main__":
    
    test_papers = [
        os.path.join("data", "papers", f) 
        for f in os.listdir("data/papers") 
        if f.endswith(".pdf")
    ]

    if not test_papers:
        print("\n!!! WARNING !!!")
        print("No PDF files found in 'data/papers/'. Please add a sample PDF (e.g., 'data/papers/sample.pdf') to test.")
        print("Skipping PDF scraper test.")
    else:
        print(f"\n--- Running PDF Scraper Test on {len(test_papers)} files ---")
        
        sample_to_test = test_papers[:5]
        
        results = []
        for path in sample_to_test:
            result = extract_pdf_sections(path)
            results.append(result)
            
        print("\n--- Summary of PDF Results ---")
        for r in results:
            print(f"File: {r.get('path')}")
            print(f"Title Guess: {r.get('title')}")
            print(f"Abstract Found (Length): {len(r.get('abstract', ''))} chars")
            print(f"Intro Found (Length): {len(r.get('introduction', ''))} chars")
            print(f"Conclusion Found (Length): {len(r.get('conclusion', ''))} chars")
            if r.get('error'):
                 print(f"ERROR: {r['error']}")
            print("-" * 30)