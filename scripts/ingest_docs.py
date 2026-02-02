import requests
import json

# Ingest test document
url = "http://localhost:8000/ingest-text"
payload = {
    "filename": "test.txt",
    "content": """This is a test document for RAG system.
Machine learning is a subset of artificial intelligence.
It focuses on creating algorithms that can learn from data.
Deep learning uses neural networks with multiple layers.
Natural language processing helps computers understand human language.
Computer vision enables machines to interpret visual information from images and videos."""
}

print("Ingesting document...")
r = requests.post(url, json=payload)
print(f"Status: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")
