from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Add route to serve index.html at root
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))

# Serve static files
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

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
    # Simple knowledge base for common questions
    question_lower = question.lower()
    
    # Knowledge base with common answers
    knowledge_base = {
        "what is ai": """Artificial Intelligence (AI) is the simulation of human intelligence by computers. Key points:

1. **Definition**: AI refers to machines or software that can perform tasks that typically require human intelligence.

2. **Types of AI**:
   - Narrow AI: Designed for specific tasks (like chatbots, image recognition)
   - General AI: Hypothetical AI with human-level intelligence

3. **Key Technologies**:
   - Machine Learning: Systems that learn from data
   - Deep Learning: Neural networks with multiple layers
   - Natural Language Processing: Understanding human language

4. **Applications**:
   - Virtual assistants (Siri, Alexa)
   - Recommendation systems (Netflix, YouTube)
   - Autonomous vehicles
   - Medical diagnosis
   - Image and speech recognition

5. **Future Impact**: AI is revolutionizing industries and will continue to play a major role in technology and society.

**In Summary**: AI enables computers to learn, reason, and make decisions autonomously, improving efficiency across many fields.""",

        "what is machine learning": """Machine Learning (ML) is a subset of AI that enables computers to learn from data without explicit programming.

1. **Core Concept**: Instead of being programmed with specific instructions, ML systems learn patterns from examples and data.

2. **How it Works**:
   - Collect training data
   - Algorithm learns patterns from data
   - Model makes predictions on new data

3. **Types of ML**:
   - Supervised Learning: Learning from labeled examples
   - Unsupervised Learning: Finding patterns in unlabeled data
   - Reinforcement Learning: Learning through rewards/penalties

4. **Common Algorithms**:
   - Decision Trees
   - Random Forests
   - Neural Networks
   - Support Vector Machines

5. **Real-World Examples**:
   - Email spam detection
   - Product recommendations
   - Facial recognition
   - Autonomous driving
   - Fraud detection

6. **Key Advantage**: Automatically improves with more data without manual reprogramming.""",

        "what is deep learning": """Deep Learning is an advanced form of Machine Learning using artificial neural networks with multiple layers.

1. **What Makes it "Deep"**: Multiple layers of interconnected neurons that process information hierarchically.

2. **Key Components**:
   - Input layer: Receives raw data
   - Hidden layers: Process and transform data
   - Output layer: Makes final prediction

3. **Advantages**:
   - Excellent for complex patterns
   - Works well with large datasets
   - Automatic feature extraction

4. **Applications**:
   - Image recognition and computer vision
   - Natural language processing
   - Speech recognition
   - Autonomous vehicles
   - Game playing (AlphaGo)

5. **Famous Architectures**:
   - Convolutional Neural Networks (CNN): For images
   - Recurrent Neural Networks (RNN): For sequences
   - Transformers: For language

6. **Challenges**: Requires lots of data and computational power.""",
    }
    
    # Check if we have a good answer for this question
    for key, answer in knowledge_base.items():
        if key in question_lower:
            return answer
    
    # Default comprehensive answer for unknown questions
    return f"""Based on the question: "{question}"

While I don't have a specific answer for this exact question, here's a general approach to solving it:

1. **Break Down the Problem**:
   - Identify the core concept being asked
   - Look for key terms and definitions
   - Understand the context

2. **Key Principles to Consider**:
   - Think about fundamental concepts related to the topic
   - Consider real-world applications
   - Look for patterns and relationships

3. **Step-by-Step Solution**:
   - Start with basic understanding
   - Build on foundational knowledge
   - Apply specific techniques relevant to the problem

4. **Verify Your Answer**:
   - Check if the answer makes sense
   - Look for consistency
   - Consider edge cases

5. **Related Topics to Explore**:
   - Broader concepts that might help
   - Similar problems and solutions
   - Practical applications

**Tip**: For more accurate answers, try asking about specific AI, Machine Learning, or Deep Learning concepts!"""

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
