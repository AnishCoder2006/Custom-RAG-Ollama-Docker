from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging
import time
import json
import asyncio

# Import from rag module
from .rag import (
    ingest_uploaded_files, 
    get_vectorstore, 
    get_rag_chain, 
    get_retriever, 
    ingest_all_documents_in_data_dir,
    clear_embedding_cache,
    get_collection_stats,
    get_embeddings,
    get_llm
)

from .models import QueryRequest, IngestResponse, IngestTextRequest, QueryResponse
from .config import DATA_DIR,CHROMA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Local RAG API - Ollama and Chroma",
    version="1.0.0",
    description="Fast RAG API with Ollama embeddings and Chroma vector store"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests
    if process_time > 5.0:
        logger.warning(f"Slow request: {request.url.path} took {process_time:.2f}s")
    
    return response

# ─── Health & Status Endpoints ───────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "RAG API is running",
        "endpoints": {
            "health": "/health",
            "ingest": "/ingest",
            "query": "/query",
            "clear": "/clear-index",
            "stats": "/stats",
            "debug": "/debug/*"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        vs = get_vectorstore()
        count = vs._collection.count() if hasattr(vs._collection, 'count') else "unknown"
        
        # Test embeddings
        embeddings = get_embeddings()
        test_emb = embeddings.embed_query("test")
        
        return {
            "status": "healthy",
            "documents": count,
            "embeddings_working": len(test_emb) > 0,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        collection_stats = get_collection_stats()
        
        return {
            "status": "success",
            "collection": collection_stats,
            "data_directory": str(DATA_DIR),
            "files_in_directory": len(list(DATA_DIR.glob("*")))
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ─── Document Ingestion Endpoints ────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_files(files: List[UploadFile] = File(...)):
    """Ingest uploaded files - optimized for speed"""
    if not files:
        raise HTTPException(400, "No files uploaded")
    
    start_time = time.time()
    saved_paths = []
    
    # Save uploaded files
    for file in files:
        filename = file.filename.replace("..", "").replace("/", "_").replace("\\", "_")
        path = DATA_DIR / filename
        
        try:
            content = await file.read()
            if len(content) > 50 * 1024 * 1024:  # 50MB limit
                raise HTTPException(400, f"File {filename} too large")
            
            with open(path, "wb") as f:
                f.write(content)
            
            saved_paths.append(str(path))
            logger.info(f"Saved file: {filename} ({len(content)} bytes)")
            
        except Exception as e:
            logger.error(f"Error saving file {filename}: {e}")
            raise HTTPException(500, f"Error saving file: {str(e)}")
    
    if not saved_paths:
        raise HTTPException(400, "No valid files to process")
    
    # Ingest files
    try:
        num_chunks = ingest_uploaded_files(saved_paths)
        
        total_time = time.time() - start_time
        logger.info(f"Ingested {len(files)} file(s) in {total_time:.2f}s → {num_chunks} chunks")
        
        return IngestResponse(
            message=f"Ingested {len(files)} file(s) in {total_time:.2f}s → {num_chunks} chunks",
            num_docs=len(files),
            num_chunks=num_chunks
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(500, f"Ingestion failed: {str(e)}")

@app.post("/ingest-text", response_model=IngestResponse)
async def ingest_text(req: IngestTextRequest):
    """Ingest plain text content"""
    start_time = time.time()
    
    if not req.content.strip():
        raise HTTPException(400, "Empty content")
    if len(req.content) > 10_000_000:  # 10MB limit
        raise HTTPException(400, "Content too large")
    
    try:
        from langchain_core.documents import Document
        from .rag import simple_chunk_text  # Import from rag module

        doc = Document(page_content=req.content, metadata={"source": req.filename})
        
        # Use efficient chunking
        chunks_text = simple_chunk_text(req.content, chunk_size=600, overlap=100)
        
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={"source": req.filename, "chunk_id": i}
            )
            chunks.append(chunk_doc)
        
        get_vectorstore().add_documents(chunks)
        
        total_time = time.time() - start_time
        logger.info(f"Ingested text '{req.filename}' in {total_time:.2f}s → {len(chunks)} chunks")
        
        return IngestResponse(
            message=f"Ingested '{req.filename}' in {total_time:.2f}s → {len(chunks)} chunks",
            num_docs=1,
            num_chunks=len(chunks)
        )
        
    except Exception as e:
        logger.error(f"Text ingest failed: {e}")
        raise HTTPException(500, f"Text ingest failed: {str(e)}")

# ─── Query Endpoint ─────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """
    Main query endpoint - optimized for speed
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question required")
    
    total_start = time.time()
    question = req.question.strip()
    logger.info(f"Query received: '{question[:50]}...' | session_id={req.session_id}")
    
    try:
        # Phase 1: Retrieve documents (with timing)
        retrieval_start = time.time()
        retriever = get_retriever()
        docs = await retriever.ainvoke(question)
        retrieval_time = time.time() - retrieval_start
        
        logger.info(f"Retrieved {len(docs)} documents in {retrieval_time:.2f}s")
        
        # Log retrieved document previews
        for i, doc in enumerate(docs[:3]):  # Log first 3
            source = doc.metadata.get("source", "Unknown")
            preview = doc.page_content[:100].replace("\n", " ")
            logger.info(f"  Doc {i+1}: {source} - '{preview}...'")
        
        # Phase 2: Format sources for response
        sources = []
        for i, doc in enumerate(docs[:5], 1):
            src = doc.metadata.get("source", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "")
            snippet = doc.page_content[:150].replace("\n", " ").strip()
            
            line = f"{i}. **{src}**"
            if chunk_id:
                line += f" [Chunk {chunk_id}]"
            line += f": {snippet}..."
            sources.append(line)
        
        # Phase 3: Generate answer (with timing)
        generation_start = time.time()
        chain_callable = get_rag_chain(req.session_id)
        answer = await chain_callable({"question": question})
        generation_time = time.time() - generation_start
        
        # Phase 4: Compile response
        total_time = time.time() - total_start
        
        logger.info(f"Query completed in {total_time:.2f}s "
                   f"(retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
        logger.info(f"Answer length: {len(answer)} chars")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            timing={
                "total": round(total_time, 2),
                "retrieval": round(retrieval_time, 2),
                "generation": round(generation_time, 2)
            }
        )
        
    except asyncio.TimeoutError:
        error_msg = "Request timeout - backend is processing slowly"
        logger.error(error_msg)
        raise HTTPException(status_code=504, detail=error_msg)
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# ─── Management Endpoints ───────────────────────────────────────────────

@app.get("/clear-index")
async def clear_index():
    """Clear the entire vector store index"""
    try:
        start_time = time.time()
        get_vectorstore().delete_collection()
        clear_embedding_cache()
        
        # Reinitialize empty collection
        from langchain_chroma import Chroma
        from .rag import get_embeddings
        _ = Chroma(
            collection_name="rag_collection",
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Index cleared in {elapsed:.2f}s")
        
        return {
            "message": "Index cleared successfully",
            "time_taken": f"{elapsed:.2f}s"
        }
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(500, f"Clear failed: {str(e)}")

@app.get("/clear-cache")
async def clear_cache():
    """Clear the embedding cache"""
    try:
        clear_embedding_cache()
        return {"message": "Embedding cache cleared"}
    except Exception as e:
        return {"error": str(e)}

# ─── Debug Endpoints ───────────────────────────────────────────────────

@app.get("/debug/collection-info")
async def debug_collection_info():
    """Debug endpoint to check collection status"""
    try:
        vs = get_vectorstore()
        collection = vs._collection
        
        # Get count
        count = collection.count() if hasattr(collection, 'count') else 0
        
        # Get sample documents
        sample_docs = []
        try:
            results = collection.get(limit=3)
            if results and 'documents' in results and results['documents']:
                for i, doc in enumerate(results['documents'][:3]):
                    sample_docs.append({
                        "id": i,
                        "preview": doc[:150] + "..." if len(doc) > 150 else doc,
                        "length": len(doc),
                        "metadata": results['metadatas'][i] if 'metadatas' in results and i < len(results['metadatas']) else {}
                    })
        except Exception as e:
            sample_docs = [f"Error fetching: {str(e)}"]
        
        return {
            "document_count": count,
            "sample_documents": sample_docs,
            "collection_name": collection.name if hasattr(collection, 'name') else "rag_collection"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/debug/test-retrieval")
async def test_retrieval(question: str = "test"):
    """Test retrieval functionality"""
    try:
        start_time = time.time()
        retriever = get_retriever()
        docs = await retriever.ainvoke(question)
        elapsed = time.time() - start_time
        
        response = {
            "question": question,
            "retrieval_time": f"{elapsed:.2f}s",
            "num_docs_retrieved": len(docs),
            "documents": []
        }
        
        for i, doc in enumerate(docs):
            response["documents"].append({
                "id": i+1,
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "content_length": len(doc.page_content),
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "Unknown")
            })
        
        return response
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/list-files")
async def debug_list_files():
    """List all files in the data directory"""
    try:
        files = list(DATA_DIR.glob("*"))
        return {
            "data_directory": str(DATA_DIR),
            "total_files": len(files),
            "files": [
                {
                    "name": f.name,
                    "size_bytes": f.stat().st_size,
                    "size_human": f"{f.stat().st_size / 1024:.1f} KB",
                    "type": "directory" if f.is_dir() else "file",
                    "extension": f.suffix if f.is_file() else ""
                }
                for f in files
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/system-status")
async def debug_system_status():
    """Comprehensive system status"""
    try:
        import psutil
        import httpx
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Ollama status
        ollama_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"http://ollama:11434/api/tags")
                ollama_status = "online" if response.status_code == 200 else "offline"
        except:
            ollama_status = "offline"
        
        # Collection stats
        collection_stats = get_collection_stats()
        
        return {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2)
            },
            "services": {
                "ollama": ollama_status,
                "chroma": "online" if 'error' not in collection_stats else "error"
            },
            "rag_system": collection_stats,
            "cache": {
                "embedding_cache_size": len(embedding_cache)
            }
        }
    except Exception as e:
        return {"error": str(e)}

# ─── Batch Operations ──────────────────────────────────────────────────

@app.post("/ingest-all")
async def ingest_all():
    """Ingest all documents from data directory"""
    try:
        start_time = time.time()
        num_chunks = ingest_all_documents_in_data_dir()
        elapsed = time.time() - start_time
        
        return {
            "message": f"Ingested {num_chunks} chunks from data directory",
            "time_taken": f"{elapsed:.2f}s",
            "chunks_ingested": num_chunks
        }
    except Exception as e:
        logger.error(f"Ingest all failed: {e}")
        raise HTTPException(500, f"Ingest all failed: {str(e)}")

@app.post("/batch-query")
async def batch_query(questions: List[str]):
    """Process multiple queries in batch"""
    try:
        results = []
        total_start = time.time()
        
        for i, question in enumerate(questions):
            question_start = time.time()
            
            try:
                retriever = get_retriever()
                docs = await retriever.ainvoke(question)
                
                chain = get_rag_chain()
                answer = await chain({"question": question})
                
                question_time = time.time() - question_start
                
                results.append({
                    "question": question,
                    "answer": answer,
                    "docs_retrieved": len(docs),
                    "time_taken": round(question_time, 2)
                })
                
                logger.info(f"Batch query {i+1}/{len(questions)} completed in {question_time:.2f}s")
                
            except Exception as e:
                results.append({
                    "question": question,
                    "error": str(e),
                    "time_taken": round(time.time() - question_start, 2)
                })
        
        total_time = time.time() - total_start
        
        return {
            "total_queries": len(questions),
            "total_time": round(total_time, 2),
            "avg_time_per_query": round(total_time / len(questions), 2),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch query failed: {e}")
        raise HTTPException(500, f"Batch query failed: {str(e)}")

# ─── Startup Event ────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("RAG API starting up...")
    
    # Test connections
    try:
        # Test vector store
        vs = get_vectorstore()
        logger.info(f"Vector store initialized: {vs._collection.name if hasattr(vs, '_collection') else 'unknown'}")
        
        # Test embeddings
        embeddings = get_embeddings()
        test_emb = embeddings.embed_query("test")
        logger.info(f"Embeddings working: {len(test_emb)}-dimensional vectors")
        
        logger.info("RAG API startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")