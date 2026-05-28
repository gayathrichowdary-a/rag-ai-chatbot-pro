import streamlit as st
import time
import tempfile
import os
import re
from dotenv import load_dotenv
from collections import defaultdict
import numpy as np

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import MergerRetriever
from langchain.schema import Document

from streamlit_pdf_viewer import pdf_viewer

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RAG AI Chatbot Pro",
    page_icon="📄",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── MAIN BACKGROUND ── */
.main,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.block-container {
    background-color: #ffffff !important;
}

.block-container {
    max-width: 900px !important;
    padding: 1.5rem 2rem 4rem 2rem !important;
}

/* ── HEADER ── */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-top: 10px;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
    margin-top: -8px;
    margin-bottom: 24px;
    letter-spacing: 0.3px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #f1f5f9 !important;
    border-right: 1px solid #e2e8f0 !important;
}

[data-testid="stSidebar"] * {
    color: #1e293b !important;
}

/* ── TABS ── */
[data-testid="stTabs"] button {
    color: #64748b !important;
    font-weight: 500;
    font-size: 14px;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: #0d1323 !important;
    border: 1.5px dashed #1e3a5f !important;
    border-radius: 14px !important;
    padding: 8px !important;
}

/* ── USER CHAT BUBBLE ── */
.chat-user {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    padding: 14px 18px;
    border-radius: 20px 20px 4px 20px;
    color: white;
    font-size: 15px;
    line-height: 1.6;
    margin: 16px 0 16px auto;
    width: fit-content;
    max-width: 75%;
    box-shadow: 0 4px 20px rgba(37,99,235,0.3);
}
/* ── AI CHAT BUBBLE ── */
.chat-ai {
    background: #f1f5f9;
    color: #1e293b;         /* dark text so it's readable on white */
    border: 1px solid #e2e8f0;
}

[data-testid="stChatMessage"] {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #1e293b !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #1e293b !important;  /* dark text on white bg */
}
            
/* ── STREAMLIT CHAT MESSAGE ── */
[data-testid="stChatMessage"] {
    background: #111c30 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 16px !important;
    padding: 14px 18px !important;
    margin: 10px 0 !important;
    color: #e2e8f0 !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.3) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #e2e8f0 !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: #0d1323 !important;
    border: 1.5px solid #1e3a5f !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
}

[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
    background: transparent !important;
}

/* ── BUTTONS ── */
[data-testid="stButton"] button {
    background: #111c30 !important;
    border: 1px solid #1e3a5f !important;
    color: #94a3b8 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stButton"] button:hover {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
    border-color: #38bdf8 !important;
}

/* ── SUCCESS / INFO BOXES ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
}

/* ── METRIC CARD ── */
.metric-card {
    background: #111c30;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #1e3a5f;
    text-align: center;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
}

/* ── DIVIDER ── */
hr {
    border-color: #1e2d45 !important;
    margin: 1rem 0 !important;
}
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class='title'>
RAG AI Chatbot Pro
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='subtitle'>
🚀 Production-Level Multi PDF AI Assistant
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}

if "active_chat" not in st.session_state:
    st.session_state.active_chat = "Chat 1"

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "reranker" not in st.session_state:
    st.session_state.reranker = None

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🧠 Workspace")

    # =====================================================
    # NEW CHAT
    # =====================================================

    if st.button("➕ New Chat", use_container_width=True):

        chat_id = f"Chat {len(st.session_state.chat_sessions)+1}"

        st.session_state.chat_sessions[chat_id] = []

        st.session_state.messages = []

        st.session_state.active_chat = chat_id

        st.rerun()

    st.divider()

    st.subheader("💬 Recent Chats")

    # =====================================================
    # CLEAR CHAT
    # =====================================================

    if st.button("🗑️ Clear Current Chat", use_container_width=True):

        st.session_state.messages = []

        st.session_state.chat_sessions[
            st.session_state.active_chat
        ] = []

        st.rerun()

    # =====================================================
    # CHAT LIST + RENAME
    # =====================================================

    for chat in list(st.session_state.chat_sessions.keys()):

        col1, col2 = st.columns([4, 1])

        with col1:

            if st.button(
                chat,
                key=f"open_{chat}",
                use_container_width=True
            ):

                st.session_state.active_chat = chat

                st.session_state.messages = (
                    st.session_state.chat_sessions[chat]
                )

                st.rerun()

        with col2:

            if st.button(
                "✏️",
                key=f"rename_{chat}"
            ):

                st.session_state.rename_chat = chat

    # =====================================================
    # RENAME CHAT INPUT
    # =====================================================

    if "rename_chat" in st.session_state:

        old_name = st.session_state.rename_chat

        new_name = st.text_input(
            "Rename Chat",
            value=old_name
        )

        if st.button("✅ Save Name"):

            st.session_state.chat_sessions[new_name] = (
                st.session_state.chat_sessions.pop(old_name)
            )

            st.session_state.active_chat = new_name

            del st.session_state.rename_chat

            st.rerun()
# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "📂 Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "💬 Chat",
    "📄 PDF Viewer",
    "📊 Analytics"
])

# =========================================================
# PDF PROCESSING
# =========================================================

if uploaded_files:

    all_docs = []

    st.session_state.llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1200
    )

    st.session_state.reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    with st.spinner("📄 Processing PDFs..."):

        for uploaded_file in uploaded_files:

            pdf_bytes = uploaded_file.getvalue()

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            )

            temp_file.write(pdf_bytes)
            temp_file.flush()

            loader = PyMuPDFLoader(temp_file.name)

            pages = loader.load()

            for i, doc in enumerate(pages):

                doc.metadata["source"] = uploaded_file.name
                doc.metadata["page"] = i + 1

                doc.page_content = (
                    f"[SOURCE: {uploaded_file.name}] "
                    f"[PAGE: {i+1}]\n\n"
                    + doc.page_content
                )

            all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    parent_docs = splitter.split_documents(all_docs)

    chunks = []

    for pid, parent in enumerate(parent_docs):

        children = sub_splitter.split_text(parent.page_content)

        for child in children:

            chunks.append(
                Document(
                    page_content=child,
                    metadata={
                        **parent.metadata,
                        "parent_id": pid
                    }
                )
            )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    faiss = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 20,
            "fetch_k": 40,
            "lambda_mult": 0.7
        }
    )

    bm25 = BM25Retriever.from_documents(chunks)

    bm25.k = 12

    st.session_state.retriever = MergerRetriever(
    retrievers=[faiss, bm25]
)

    st.success(f"✅ Loaded Successfully: {len(chunks)} chunks")

# =========================================================
# CHAT TAB
# =========================================================

with tab1:

    chat_container = st.container()

    # DISPLAY OLD MESSAGES
    with chat_container:

        for msg in st.session_state.messages:

            if msg["role"] == "user":

                st.markdown(
                    f"""
                    <div class='chat-user'>
                    🧑 {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class='chat-ai'>
                    🤖 {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # CHAT INPUT
    if uploaded_files and st.session_state.retriever:

        query = st.chat_input(
            "Ask anything from your PDFs..."
        )

        if query:

            # =========================================================
            # SHOW USER QUESTION IMMEDIATELY
            # =========================================================

            st.session_state.messages.append({
                "role": "user",
                "content": query
            })

            with chat_container:

                st.markdown(
                    f"""
                    <div class='chat-user'>
                    🧑 {query}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            retriever = st.session_state.retriever
            llm = st.session_state.llm
            reranker = st.session_state.reranker

            # =========================================================
            # CHAT HISTORY
            # =========================================================

            chat_history = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in st.session_state.messages[-6:]
            )

            # =========================================================
            # QUERY BOOSTING
            # =========================================================

            boost_terms = []

            q = query.lower()

            if "cgpa" in q:

                boost_terms.extend([
                    "cgpa",
                    "education",
                    "academic"
                ])

            if "internship" in q:

                boost_terms.extend([
                    "internship",
                    "experience",
                    "training"
                ])

            if "project" in q:

                boost_terms.extend([
                    "project",
                    "system",
                    "implementation"
                ])

            if (
                "generative ai" in q
                or "gen ai" in q
                or "llm" in q
            ):

                boost_terms.extend([
                    "generative ai",
                    "genai",
                    "llm",
                    "rag",
                    "ai projects",
                    "streamlit",
                    "medical report analysis",
                    "fraud detection",
                    "ai generated content"
                ])

            if "best" in q and "generative ai" in q:

                boost_terms.extend([
                    "generative ai",
                    "llm",
                    "rag",
                    "ai projects",
                    "streamlit",
                    "medical report analysis",
                    "genai"
                ])

            enhanced_query = (
                query + " " + " ".join(boost_terms)
            )

            # =========================================================
            # DETECT QUERY TYPE
            # =========================================================

            query_type = "normal"

            if (
                "summary" in q
                or "summarize" in q
                or "overview" in q
            ):

                query_type = "summary"

            # =========================================================
            # RETRIEVAL
            # =========================================================

            retrieved_docs = retriever.get_relevant_documents(
                enhanced_query
            )

            pairs = [
                (query, d.page_content)
                for d in retrieved_docs
            ]

            scores = reranker.predict(pairs)

            ranked = sorted(
                zip(scores, retrieved_docs),
                key=lambda x: x[0],
                reverse=True
            )

            top_docs = []

            seen_parent_ids = set()

            for score, doc in ranked:

                pid = doc.metadata.get("parent_id")

                if pid not in seen_parent_ids:

                    top_docs.append(doc)

                    seen_parent_ids.add(pid)

                if len(top_docs) >= 6:
                    break

            top_scores = [s for s, _ in ranked[:5]]

            # =========================================================
            # CONTEXT BUILDING
            # =========================================================

            grouped_context = defaultdict(list)

            for doc in top_docs:

                src = doc.metadata.get("source")
                page = doc.metadata.get("page")

                grouped_context[src].append(
                    f"[PAGE {page}]\n{doc.page_content}"
                )

            context_text = ""

            for src, chunks_text in grouped_context.items():

                context_text += f"\n\n===== {src} =====\n"

                if query_type == "summary":

                    context_text += "\n".join(
                        chunks_text[:6]
                    )

                else:

                    context_text += "\n".join(
                        chunks_text[:3]
                    )

            # =========================================================
            # SYSTEM PROMPT
            # =========================================================

            SYSTEM_PROMPT = """
You are an enterprise-grade RAG AI assistant.

STRICT RULES:
1. Answer ONLY using provided PDF context
2. Never hallucinate
3. Mention exact PDF names
4. Mention page numbers whenever possible
5. If answer not found say:
   "The uploaded PDFs do not contain this information."
6. Compare PDFs carefully
7. Use bullet points when needed
8. Keep answers professional and concise
9. Never invent internships, projects, CGPA, or certifications
"""

            prompt = f"""
{SYSTEM_PROMPT}

================ CONTEXT ================
{context_text}

================ CHAT HISTORY ================
{chat_history}

================ QUESTION ================
{query}
"""

            # =========================================================
            # LLM RESPONSE
            # =========================================================

            with st.spinner("🤖 AI Thinking..."):

                response = llm.invoke(prompt)

                answer = response.content
            answer = re.sub(r'<[^>]+>', '', answer)  # strip any HTML tags
    

            # =========================================================
            # CONFIDENCE SCORE
            # =========================================================

            arr = np.array(top_scores)

            if (
                len(arr) > 1
                and np.max(arr) != np.min(arr)
            ):

                confidence = (
                    (np.mean(arr) - np.min(arr))
                    /
                    (np.max(arr) - np.min(arr) + 1e-6)
                )

            else:

                confidence = 0.6

            confidence = int(confidence * 100)

            warning_text = ""

            if confidence < 35:

                warning_text = """
⚠️ Low confidence answer.
Limited matching context found.
"""

            final_answer = f"""
{answer}

{warning_text}

📊 Confidence Score: {confidence}%
"""

            # =========================================================
            # DISPLAY AI RESPONSE
            # =========================================================
            with chat_container:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    out = ""
                    for word in final_answer.split():
                        out += word + " "
                        time.sleep(0.002)
                        placeholder.markdown(out + "▌", unsafe_allow_html=False)
                    placeholder.markdown(out, unsafe_allow_html=False)
           

            # =========================================================
            # SAVE CHAT
            # =========================================================

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer
            })

            st.session_state.chat_sessions[
                st.session_state.active_chat
            ] = st.session_state.messages


# =========================================================
# PDF VIEWER
# =========================================================

with tab2:

    if uploaded_files:

        selected_pdf = st.selectbox(
            "Select PDF",
            [f.name for f in uploaded_files]
        )

        for file in uploaded_files:

            if file.name == selected_pdf:

                pdf_viewer(
                    file.getvalue(),
                    width=1000,
                    height=700
                )

# =========================================================
# ANALYTICS
# =========================================================

with tab3:

    if uploaded_files:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
            <div class='metric-card'>
            <h2>{len(uploaded_files)}</h2>
            <p>Uploaded PDFs</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class='metric-card'>
            <h2>{len(st.session_state.messages)}</h2>
            <p>Total Messages</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:

            st.markdown(f"""
            <div class='metric-card'>
            <h2>RAG + LLM</h2>
            <p>Enterprise AI Pipeline</p>
            </div>
            """, unsafe_allow_html=True)