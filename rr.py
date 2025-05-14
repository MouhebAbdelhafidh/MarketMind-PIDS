from smolagents import GoogleSearchTool
from typing import Dict, TypedDict, List, Optional, Any
from langgraph.graph import StateGraph
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
import chromadb
import torch
from diffusers import StableDiffusionPipeline
import os
from datetime import datetime
import uuid
import json
from langchain_core.messages import HumanMessage
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Configuration
os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"

# Local directory setup
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir("chroma_db")
ensure_dir("memory_db")
ensure_dir("Data")

# Initialize Stable Diffusion with CPU/GPU compatibility
if torch.cuda.is_available():
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16
    ).to("cuda")
else:
    sd_pipe = None

# System prompt with Mistral formatting
system_prompt = """<<SYS>>
**Role**: Expert Digital Marketing Strategist Assistant
**Core Capabilities**:
- Generating detailed, step-by-step marketing strategies
- Creating customized campaign blueprints
- Analyzing market/audience data for strategy optimization
- Providing measurable KPIs for all recommendations
- Offering tiered solutions for different budget levels
<</SYS>>"""

# Ollama LLM initialization
query_wrapper_prompt = PromptTemplate("[INST]{query_str}[/INST]")
llm = Ollama(model="mistral", system_prompt=system_prompt, temperature=0.7, context_window=8192, request_timeout=300, base_url="http://localhost:11434", query_wrapper_prompt=query_wrapper_prompt)

# Load documents and create index
try:
    documents = SimpleDirectoryReader("Data").load_data()
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("my_collection"))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
except Exception as e:
    print(f"Document loading or indexing error: {str(e)}")

# Search documents
def search_documents(query: str) -> str:
    try:
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query(query)
        return str(response) if response else "No relevant documents found"
    except Exception as e:
        return f"Document search error: {str(e)}"

# Execute search
def execute_search(question: str) -> str:
    try:
        result = search_documents(question)
        if "No relevant documents found" not in result:
            return result
    except Exception as e:
        if "timed out" in str(e):
            response = llm.complete(f"[INST]{question}[/INST]")
            return response.text.strip() if response and response.text else "No response generated due to timeout."
        return f"Search error: {str(e)}"

    response = llm.complete(f"[INST]{question}[/INST]")
    return response.text.strip() if response and response.text else "No response generated."

# Run the assistant
def run_agent():
    print("Marketing AI Assistant - Local Version (Ollama)")
    while True:
        user_input = input("\nQuestion (or 'exit' to quit): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        print("\nMarketing Answer:\n", execute_search(user_input))

if __name__ == "__main__":
    run_agent()
