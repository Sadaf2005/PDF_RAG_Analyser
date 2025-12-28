
import os
from dotenv import load_dotenv


load_dotenv()

try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage
    
    
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant") 

    print(f"Attempting to connect to Groq with model: {GROQ_MODEL}")

  
    llm = ChatGroq(model_name=GROQ_MODEL)

    messages = [
        SystemMessage(content="You are a helpful assistant specialized in explaining technical concepts concisely."),
        HumanMessage(content="Summarize: Transformers use attention...")
    ]
    
    resp = llm.invoke(messages)
    
    print("\n--- LLM Response ---")
    print(resp.content)
    print("--------------------\n")

except ImportError as e:
    print(f"Error: {e}. Did you install all dependencies from requirements.txt?")
except Exception as e:
    
    print(f"An error occurred during the LLM call. Check your GROQ_API_KEY and model name: {e}")