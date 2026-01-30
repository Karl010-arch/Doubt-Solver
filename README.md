# 🎓 Anonymous AI Doubt Solver

A hackathon project that helps students ask doubts anonymously, prevents duplicate questions using AI similarity, and ensures no doubt goes unanswered using AI support.

## 🚀 Features
- Anonymous doubt posting
- AI duplicate detection
- AI-generated explanations
- Simple web interface

## 🛠 Tech Stack
- FastAPI
- HTML + JavaScript
- Sentence Transformers
- Cosine Similarity

## ▶️ Run Locally
```bash
pip install fastapi uvicorn sentence-transformers scikit-learn
uvicorn backend.main:app --reload
