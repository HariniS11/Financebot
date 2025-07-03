# 🤖 Financebot using RAG (LangChain + OpenAI + FAISS)

## 🔍 Overview
This is a **Finance Question-Answering Chatbot** built using **LangChain**, **OpenAI (LLM & Embeddings)**, and **FAISS** for vector search. It reads finance-related articles from URLs, splits the content into chunks, embeds them, and stores them in a vector database. Using **Retrieval-Augmented Generation (RAG)**, it allows users to ask questions about the content through an interactive **Streamlit** app.

---

## 🚀 Features
- 📄 Accepts article URLs and fetches content automatically
- ✂️ Splits content into semantically meaningful chunks
- 🧠 Embeds chunks using OpenAI Embeddings
- 📦 Stores vectors in FAISS for fast retrieval
- 🗣️ Uses OpenAI LLM to generate answers from context
- 🌐 Streamlit frontend for user interaction

---

## 🛠️ Tech Stack

| Component      | Tool/Library                            |
|----------------|-----------------------------------------|
| LLM            | `OpenAI` (via LangChain)                |
| Embeddings     | `OpenAIEmbeddings`                      |
| Vector DB      | `FAISS`                                 |
| Framework      | `LangChain`                             |
| UI             | `Streamlit`                             |
| Loader         | `UnstructuredURLLoader` (URL-based)     |

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/finance-rag-chatbot.git
cd finance-rag-chatbot
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
