🤖 RAG Document Assistant
An AI-powered document Q&A system that allows you to upload documents and ask questions about their content. Built with LangChain, FastAPI, Streamlit, and Ollama.

https://docs/architecture.png

✨ Features
📄 Document Upload: Upload PDF and TXT files

🤖 AI-Powered Q&A: Ask questions in natural language

🔍 Semantic Search: Find relevant information with source citations

🐳 Dockerized: Easy deployment with Docker Compose

🔒 Privacy-First: 100% local processing, no data leaves your system

⚡ Fast Responses: Optimized for speed with caching and batching

💬 Conversation Memory: Maintains context across questions

🏗️ Architecture
text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│    FastAPI      │────▶│    LangChain    │
│    Frontend     │◀────│    Backend      │◀────│     RAG         │
│   (Port 8501)   │     │   (Port 8000)   │     │    Pipeline     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                           │
                       ┌──────┴──────┐            ┌──────┴──────┐
                       │   Ollama    │            │   ChromaDB  │
                       │    LLM      │            │   Vector    │
                       │ (Port 11434)│            │   Database  │
                       └─────────────┘            └─────────────┘
                              │                           │
                       ┌──────┴──────┐            ┌──────┴──────┐
                       │    Redis     │            │   Data      │
                       │   Memory     │            │  Directory  │
                       │ (Port 6379)  │            └─────────────┘
                       └─────────────┘
🚀 Quick Start
Prerequisites
Docker and Docker Compose

Git

Installation
Clone the repository

bash
git clone https://github.com/yourusername/rag-document-assistant.git
cd rag-document-assistant
Start the services

bash
# Using Docker Compose
docker-compose up -d

# Or using the provided script
chmod +x scripts/start.sh
./scripts/start.sh
