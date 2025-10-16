# 🇲🇦 Assistant Administratif Marocain

A **Streamlit-based AI assistant** designed to help users navigate Moroccan administrative procedures.  
This project leverages **OpenAI embeddings and LangChain** for document search and question-answering.

---

## 🗂️ Features

- Ask questions about Moroccan administrative operations (e.g., CNIE renewal, document requirements, fees).  
- Context-aware responses using vector search over a JSON dataset of official documents.  
- Chat interface similar to ChatGPT

---

## ⚙️ Tech Stack

- **Python 3.12+**
- **Streamlit** – for the interactive web app
- **LangChain** – for document embeddings and similarity search
- **FAISS** – vector store for fast similarity retrieval
- **OpenAI** – GPT models (`gpt-4o-mini` for responses, `text-embedding-3-small` for embeddings)
- **dotenv** – to load API keys securely
