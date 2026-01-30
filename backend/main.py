from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import os

app = FastAPI(title="Anonymous AI Doubt Solver")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend - mount after all API routes
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")

model = SentenceTransformer("all-MiniLM-L6-v2")

doubts_db = []
answers_db = []

class Doubt(BaseModel):
    question: str
    subject: str

class Answer(BaseModel):
    doubt_id: str
    content: str
    is_ai: bool = False

def find_similar(embedding):
    for doubt in doubts_db:
        sim = cosine_similarity([embedding], [doubt["embedding"]])[0][0]
        if sim > 0.8:
            return doubt
    return None

def generate_ai_answer(question):
    return f"""
Step-by-step explanation:

1. Understand the question.
2. Apply the basic concept.
3. Solve it clearly.

Example:
This explains the doubt: {question}

(AI Generated Answer)
"""

@app.post("/doubts")
def post_doubt(doubt: Doubt):
    embedding = model.encode(doubt.question)
    similar = find_similar(embedding)

    if similar:
        return {
            "status": "duplicate",
            "existing_doubt": similar
        }

    doubt_data = {
        "id": str(uuid.uuid4()),
        "question": doubt.question,
        "subject": doubt.subject,
        "embedding": embedding.tolist()
    }
    doubts_db.append(doubt_data)
    return doubt_data

@app.get("/doubts")
def get_doubts():
    return doubts_db

@app.post("/answers")
def post_answer(answer: Answer):
    answer_data = {
        "id": str(uuid.uuid4()),
        "doubt_id": answer.doubt_id,
        "content": answer.content,
        "votes": 0,
        "is_ai": answer.is_ai
    }
    answers_db.append(answer_data)
    return answer_data

@app.get("/answers/{doubt_id}")
def get_answers(doubt_id: str):
    return [a for a in answers_db if a["doubt_id"] == doubt_id]

@app.post("/auto-answer/{doubt_id}")
def auto_answer(doubt_id: str):
    for doubt in doubts_db:
        if doubt["id"] == doubt_id:
            ai_answer = {
                "id": str(uuid.uuid4()),
                "doubt_id": doubt_id,
                "content": generate_ai_answer(doubt["question"]),
                "votes": 0,
                "is_ai": True
            }
            answers_db.append(ai_answer)
            return ai_answer
    return {"error": "Doubt not found"}
