from typing import List, Optional, Callable, Dict, Any
import os
import asyncio
import logging
import time
import hashlib

logger = logging.getLogger(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

from .config import (
    DATA_DIR,
    CHROMA_DIR,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    K_RETRIEVE,
)

# ─── Simple Cache for Frequent Queries ─────────────────────────────────────
embedding_cache = {}

# ─── Lazy-loaded singletons ────────────────────────────────────────────────

_embedding = None
_llm = None
_vectorstore = None
_retriever = None

class SimpleOllamaEmbeddings:
    """Simple and reliable embeddings wrapper"""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = 30.0
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        import httpx
        import json
        
        if not texts:
            return []
        
        # Clean texts
        texts = [str(t).strip() for t in texts if str(t).strip()]
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} documents...")
        
        try:
            # Ollama expects: {"model": "name", "input": "text"} or {"model": "name", "input": ["text1", "text2"]}
            payload = {
                "model": self.model,
                "input": texts if len(texts) > 1 else texts[0]
            }
            
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # DEBUG: Log what we received
            logger.debug(f"Embedding response: {json.dumps(data)[:200]}...")
            
            # Handle response - Ollama returns different formats
            if isinstance(data, list):
                # Direct list of embeddings
                return data
            elif isinstance(data, dict):
                if "embeddings" in data:
                    # List of embeddings
                    return data["embeddings"]
                elif "embedding" in data:
                    # Single embedding
                    emb = data["embedding"]
                    if isinstance(emb, list):
                        if emb and isinstance(emb[0], (int, float)):
                            return [emb]  # Wrap single embedding in list
                        else:
                            return emb
                    else:
                        # Should be a list, but handle just in case
                        return [[emb]]
                else:
                    # Try to find any list in the response
                    for key, value in data.items():
                        if isinstance(value, list) and value:
                            if isinstance(value[0], (int, float)):
                                return [value]
                            elif isinstance(value[0], list):
                                return value
            return []
            
        except Exception as e:
            logger.error(f"Document embedding failed: {e}")
            # Return empty embeddings to avoid crashing
            return [[] for _ in range(len(texts))]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed multiple documents"""
        import httpx
        import json
        
        if not texts:
            return []
        
        texts = [str(t).strip() for t in texts if str(t).strip()]
        if not texts:
            return []
        
        logger.info(f"Async embedding {len(texts)} documents")
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "model": self.model,
                    "input": texts if len(texts) > 1 else texts[0]
                }
                
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                logger.debug(f"Async embedding response: {json.dumps(data)[:200]}...")
                
                # Same parsing logic as sync version
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    if "embeddings" in data:
                        return data["embeddings"]
                    elif "embedding" in data:
                        emb = data["embedding"]
                        if isinstance(emb, list):
                            if emb and isinstance(emb[0], (int, float)):
                                return [emb]
                            else:
                                return emb
                        else:
                            return [[emb]]
                    else:
                        for key, value in data.items():
                            if isinstance(value, list) and value:
                                if isinstance(value[0], (int, float)):
                                    return [value]
                                elif isinstance(value[0], list):
                                    return value
                return []
                
        except Exception as e:
            logger.error(f"Async document embedding failed: {e}")
            return [[] for _ in range(len(texts))]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query - SIMPLE VERSION"""
        text = str(text).strip()
        if not text:
            return []
        
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in embedding_cache:
            logger.info(f"Using cached embedding for query: {text[:50]}...")
            return embedding_cache[cache_key]
        
        # Get embeddings for this single query
        embeddings = self.embed_documents([text])
        
        if embeddings and len(embeddings) > 0 and embeddings[0]:
            embedding_cache[cache_key] = embeddings[0]
            return embeddings[0]
        
        # Return empty embedding if failed
        logger.error(f"Failed to get embedding for query: {text[:50]}...")
        return []
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query"""
        text = str(text).strip()
        if not text:
            return []
        
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in embedding_cache:
            logger.info(f"Using cached async embedding for query: {text[:50]}...")
            return embedding_cache[cache_key]
        
        embeddings = await self.aembed_documents([text])
        
        if embeddings and len(embeddings) > 0 and embeddings[0]:
            embedding_cache[cache_key] = embeddings[0]
            return embeddings[0]
        
        logger.error(f"Failed to get async embedding for query: {text[:50]}...")
        return []


def get_embeddings():
    global _embedding
    if _embedding is None:
        _embedding = SimpleOllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=EMBEDDING_MODEL,
        )
    return _embedding


def get_llm():
    global _llm
    if _llm is None:
        from langchain_ollama import ChatOllama
        _llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=0.3,
            num_predict=512,
        )
    return _llm


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma
        try:
            # Try to load existing collection
            _vectorstore = Chroma(
                collection_name="rag_collection",
                embedding_function=get_embeddings(),
                persist_directory=str(CHROMA_DIR),
            )
            logger.info(f"Loaded existing Chroma collection")
        except Exception as e:
            logger.warning(f"Could not load existing collection: {e}. Creating new one.")
            # Create new collection
            _vectorstore = Chroma(
                collection_name="rag_collection",
                embedding_function=get_embeddings(),
                persist_directory=str(CHROMA_DIR),
            )
            logger.info(f"Created new Chroma collection")
    return _vectorstore


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs={"k": K_RETRIEVE},
        )
    return _retriever


def get_rag_chain(session_id: Optional[str] = None):
    """Get RAG chain"""
    llm = get_llm()
    retriever = get_retriever()

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on this context:

        Context:
        {context}

        Question: {question}

        Answer:"""
    )

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            if content:
                formatted.append(f"[Doc {i+1}]: {content}")
        return "\n\n".join(formatted)

    base_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    async def async_chain(inputs: Dict[str, Any]) -> str:
        return await base_chain.ainvoke(inputs)
    
    return async_chain


# ─── Simple Chunking ────────────────────────────────────────────────────

def simple_chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """Simple text chunking"""
    if not text or len(text.strip()) == 0:
        return []
    
    if len(text) <= chunk_size:
        return [text.strip()]
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1
        
        if current_length + word_length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            
            # Start new chunk with overlap
            overlap_words = int(len(current_chunk) * (overlap / 100))
            current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
            current_length = sum(len(w) + 1 for w in current_chunk)
        
        current_chunk.append(word)
        current_length += word_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    
    return chunks


# ─── Simple Ingestion ──────────────────────────────────────────────────

def ingest_uploaded_files(file_paths: List[str]) -> int:
    """Simple file ingestion"""
    vectorstore = get_vectorstore()
    all_chunks = []
    
    logger.info(f"Ingesting {len(file_paths)} files")
    
    for path_str in file_paths:
        path_str = str(path_str)
        
        try:
            if path_str.lower().endswith('.pdf'):
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(path_str)
                documents = loader.load()
                logger.info(f"Loaded PDF: {len(documents)} pages")
            else:
                with open(path_str, 'r', encoding='utf-8') as f:
                    content = f.read()
                from langchain_core.documents import Document
                documents = [Document(page_content=content, metadata={"source": path_str})]
                logger.info(f"Loaded text file: {len(content)} chars")
            
            for doc in documents:
                content = doc.page_content
                if not content or len(content.strip()) < 10:
                    continue
                
                metadata = {"source": path_str}
                if hasattr(doc, 'metadata'):
                    metadata.update(doc.metadata)
                
                # Simple chunking
                chunks = simple_chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
                
                for i, chunk_text in enumerate(chunks):
                    if len(chunk_text.strip()) < 20:
                        continue
                    
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = len(all_chunks)
                    
                    from langchain_core.documents import Document
                    chunk_doc = Document(page_content=chunk_text, metadata=chunk_metadata)
                    all_chunks.append(chunk_doc)
            
        except Exception as e:
            logger.error(f"Error processing {path_str}: {e}")
            continue
    
    if all_chunks:
        vectorstore.add_documents(all_chunks)
        logger.info(f"Ingested {len(all_chunks)} chunks")
    
    return len(all_chunks)


def ingest_all_documents_in_data_dir() -> int:
    """Ingest all documents from data directory"""
    import glob
    
    pdf_files = glob.glob(str(DATA_DIR / "**" / "*.pdf"), recursive=True)
    text_files = glob.glob(str(DATA_DIR / "**" / "*.txt"), recursive=True)
    
    all_files = pdf_files + text_files
    
    if not all_files:
        logger.warning("No files found to ingest")
        return 0
    
    return ingest_uploaded_files(all_files)


def clear_embedding_cache():
    """Clear the embedding cache"""
    global embedding_cache
    embedding_cache.clear()
    logger.info("Embedding cache cleared")


def get_collection_stats():
    """Get collection statistics"""
    try:
        vs = get_vectorstore()
        collection = vs._collection
        
        count = collection.count() if hasattr(collection, 'count') else 0
        
        return {
            "document_count": count,
            "cache_size": len(embedding_cache)
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}


# Export all functions
__all__ = [
    'ingest_uploaded_files',
    'get_vectorstore',
    'get_rag_chain',
    'get_retriever',
    'ingest_all_documents_in_data_dir',
    'clear_embedding_cache',
    'get_collection_stats',
    'get_embeddings',
    'get_llm',
    'simple_chunk_text',
]