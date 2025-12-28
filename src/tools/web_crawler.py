import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
import json
from datetime import datetime

# --- Configuration ---
RAW_DATA_DIR = "data/raw_web"
os.makedirs(RAW_DATA_DIR, exist_ok=True)
USER_AGENT = "rag-bot/1.0 (Academic Project)"

# --- Core Function ---
def fetch_page_text(url: str, timeout: int = 15) -> dict:
    """
    Fetches a web page, cleans the content, and extracts metadata.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        A dictionary with url, title, text, and other metadata.
    """
    print(f"Fetching: {url}")
    try:
        
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status() 
        
        
        soup = BeautifulSoup(r.text, "html.parser")
        
       
        for s in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            s.decompose()
            
        
        title = soup.title.string.strip() if soup.title and soup.title.string else urlparse(url).netloc
        
       
        text_elements = soup.find_all(['p', 'li', 'h1', 'h2', 'h3', 'blockquote'])
        
        text = " ".join(e.get_text(separator=" ", strip=True) for e in text_elements if e.get_text(strip=True))
        
        cleaned_text = " ".join(text.split()).strip()

        if not cleaned_text:
             cleaned_text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
             cleaned_text = " ".join(cleaned_text.split()).strip()

        metadata = {
            "url": url,
            "title": title,
            "date_fetched": datetime.now().isoformat(),
            "source": "web_page",
            "text": cleaned_text
        }
        
        doc_id = title.replace(" ", "_").replace("/", "-").lower()[:50] + "_" + str(hash(url))[:8]
        
        with open(os.path.join(RAW_DATA_DIR, f"{doc_id}_raw.html"), "w", encoding="utf-8") as f:
            f.write(r.text)
            
        with open(os.path.join(RAW_DATA_DIR, f"{doc_id}_processed.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return {"url": url, "title": "ERROR", "text": f"Failed to fetch: {e}", "source": "web_page"}
    except Exception as e:
        print(f"An unexpected error occurred for {url}: {e}")
        return {"url": url, "title": "ERROR", "text": f"Unexpected error: {e}", "source": "web_page"}

# --- Test function ---
if __name__ == "__main__":
    sample_urls = [
        "https://en.wikipedia.org/wiki/Ontology_(information_science)",
        "https://medium.com/@cassihunt/semantic-model-vs-ontology-vs-knowledge-graph-untangling-the-latest-data-modeling-terminology-12ce7506b455",
        "https://www.sciencedirect.com/science/article/pii/S0169023X24000491" 
    ]
    
    print("\n--- Running Web Crawler ---")
    results = [fetch_page_text(url) for url in sample_urls]
    
    print("\n--- Summary of Results ---")
    for r in results:
        print(f"URL: {r['url']}")
        print(f"Title: {r['title']}")
        print(f"Text length: {len(r['text'])} characters")
        print("-" * 20)