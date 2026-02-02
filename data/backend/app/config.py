# config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Paths
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "phi3:mini")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# RAG Settings - Optimized
CHUNK_SIZE = 600  # Good for resumes
CHUNK_OVERLAP = 80
K_RETRIEVE = 1