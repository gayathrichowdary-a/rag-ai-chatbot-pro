# rag-ai-chatbot-pro

# 🚀 RAG AI Chatbot Pro

Production-Level Multi PDF AI Assistant powered by RAG (Retrieval-Augmented Generation), Groq LLM, FAISS Vector Search, and Streamlit.

---

## 🌐 Live Demo

👉 https://rag-ai-chatbot-pro-fojjmbnydk6sqkxqtudnbb.streamlit.app/

---

# ✨ Features

* 📄 Multi PDF Upload
* 🧠 RAG-based AI Question Answering
* ⚡ Groq LLM Integration
* 🔍 Hybrid Retrieval (FAISS + BM25)
* 🎯 Cross-Encoder Re-ranking
* 📊 Confidence Score Generation
* 💬 Multi Chat Sessions
* ✏️ Rename Chat Feature
* 🗑️ Clear Chat Feature
* 📑 PDF Viewer
* 📈 Analytics Dashboard
* 🎨 Modern Professional UI

---

# 🛠️ Tech Stack

## Frontend

* Streamlit

## AI / ML

* LangChain
* FAISS
* Sentence Transformers
* Cross Encoder
* BM25 Retriever
* Groq LLM

## Backend

* Python

---

# 📸 Application Screenshots

## 🏠 Home Page

![Home](assets/home.pngg.png)

---

## 📂 PDF Upload System

![Upload](assets/upload.pngg.png)

---

## 💬 AI Chat System

![Chat](assets/chat.pngg.png)

---

## 📊 Analytics Dashboard

![Analytics](assets/analytics.pngg.png)

---

# ⚙️ Installation

```bash
git clone https://github.com/gayathrichowdary-a/rag-ai-chatbot-pro.git

cd rag-ai-chatbot-pro

pip install -r requirements.txt

streamlit run app.py
```

---

# 🔐 Environment Variables

Create `.streamlit/secrets.toml`

```toml
GROQ_API_KEY="your_api_key"
```

---

# 🧠 How It Works

1. User uploads PDFs
2. PDFs are chunked using LangChain
3. FAISS + BM25 retrieve relevant chunks
4. Cross Encoder reranks results
5. Groq LLM generates final answer
6. Confidence score is calculated
7. Response displayed in Streamlit UI

---

# 🚀 Future Improvements

* Voice Chat Integration
* OCR for scanned PDFs
* Multi-user Authentication
* Chat Export Feature
* Citation Highlighting
* Advanced Memory System

---

# 👩‍💻 Developer

## A. Gayathri

Generative AI Developer | RAG Engineer | AI Enthusiast

---

# ⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork the project
📢 Share it on LinkedIn
