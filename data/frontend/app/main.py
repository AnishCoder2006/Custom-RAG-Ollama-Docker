import streamlit as st
import requests
import time
import uuid
from datetime import datetime

# ─── Config ────────────────────────────────────────────────────────────────
API_BASE_URL = "http://fastapi-backend:8000"  # Docker service name 

st.set_page_config(
    page_title="📚 Local RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Session State ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "processing" not in st.session_state:
    st.session_state.processing = False
if "backend_status" not in st.session_state:
    st.session_state.backend_status = None

# ─── Custom CSS (your dark theme) ──────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main .block-container { padding-top: 2rem; background-color: #0e1117; }
    h1, h2, h3, p, div, span, label { color: #f0f2f6 !important; }
    .stChatMessage { padding: 0.5rem 0; }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 4px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .assistant-message {
        background: #1e2229 !important;
        color: #f0f2f6 !important;
        border: 1px solid #4b5563 !important;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 4px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    .source-card {
        background: #1a1d24 !important;
        border-left: 4px solid #667eea !important;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 0.85em;
        color: #d1d5db !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ───────────────────────────────────────────────────────────────
def check_backend_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        st.session_state.backend_status = r.status_code == 200
        return st.session_state.backend_status
    except:
        st.session_state.backend_status = False
        return False

def format_time():
    return datetime.now().strftime("%H:%M")

# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h1 style="color: white;">📚 Local RAG Chat</h1>', unsafe_allow_html=True)
    st.markdown("---")

    backend_online = check_backend_health()
    status_class = "status-online" if backend_online else "status-offline"
    status_text = "Backend Connected" if backend_online else "Backend Offline"
    st.markdown(f"""
    <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
        <div class='status-indicator {status_class}'></div>
        <strong>{status_text}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    st.markdown("---")

    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and st.button("🚀 Ingest Documents", type="primary", disabled=not backend_online):
        st.session_state.processing = True
        try:
            files_to_send = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
            r = requests.post(f"{API_BASE_URL}/ingest", files=files_to_send, timeout=300)
            r.raise_for_status()
            result = r.json()
            st.session_state.ingested_files = [f.name for f in uploaded_files]
            st.success(result.get("message", "Ingestion successful!"))
            st.balloons()
        except Exception as e:
            st.error(f"Ingestion failed: {str(e)}")
        finally:
            st.session_state.processing = False

    if st.session_state.ingested_files:
        st.info(f"Indexed: {len(st.session_state.ingested_files)} files")
        if st.button("🗑️ Clear Knowledge Base"):
            try:
                requests.get(f"{API_BASE_URL}/clear-index")
                st.session_state.ingested_files = []
                st.session_state.messages = []
                st.success("Cleared!")
                st.rerun()
            except:
                st.error("Clear failed")

# ─── Main Chat ─────────────────────────────────────────────────────────────
st.title("💬 Chat with Your Documents")

chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
            css_class = "user-message" if role == "user" else "assistant-message"
            st.markdown(f'<div class="{css_class}">{msg["content"]}</div>', unsafe_allow_html=True)
            if "sources" in msg and msg["sources"]:
                with st.expander("📌 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f'<div class="source-card">{src}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)

    with chat_container:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    r = requests.post(
                        f"{API_BASE_URL}/query",
                        json={
                            "question": prompt,
                            "session_id": st.session_state.session_id
                        },
                        timeout=120
                    )
                    r.raise_for_status()
                    result = r.json()

                    answer = result.get("answer", "No answer received.")
                    sources = result.get("sources", [])

                    st.markdown(f'<div class="assistant-message">{answer}</div>', unsafe_allow_html=True)

                    if sources:
                        with st.expander("📌 Sources", expanded=False):
                            for src in sources:
                                st.markdown(f'<div class="source-card">{src}</div>', unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except requests.exceptions.RequestException as e:
                    error_msg = f"Backend error: {str(e)}"
                    st.markdown(f'<div class="assistant-message">{error_msg}</div>', unsafe_allow_html=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })