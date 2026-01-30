from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import os
import httpx
import asyncio

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

async def generate_ai_answer(question: str) -> str:
    """Generate AI answer using Hugging Face API (free tier)"""
    try:
        # Using Hugging Face's free inference API
        # Model: mistral-7b (fast and free)
        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        
        # Create a system prompt for educational answers
        prompt = f"""You are an expert tutor helping students understand concepts clearly and concisely.
Question from student: {question}

Provide a clear, concise answer that:
1. Defines the concept simply
2. Explains key points in 2-3 sentences
3. Gives a real-world example
4. Mentions applications or related concepts

Keep the answer under 300 words and easy to understand."""

        # Make request to Hugging Face (no authentication needed for basic usage)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                json={"inputs": prompt},
                headers={"User-Agent": "Doubt-Solver/1.0"}
            )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                answer_text = result[0].get("generated_text", "").strip()
                # Extract only the answer part (remove the prompt)
                if "Provide a clear" in answer_text:
                    answer_text = answer_text.split("Provide a clear")[1].strip()
                if answer_text:
                    return answer_text
    except Exception as e:
        print(f"Error calling Hugging Face API: {e}")
    
    # Fallback: Return a helpful answer using the knowledge base
    question_lower = question.lower()
    
    # Simple fallback knowledge base for quick responses
    fallback_answers = {
        "vector": "A vector is a mathematical object with magnitude and direction. Examples: velocity, force. Vectors are fundamental in physics, graphics, and machine learning for representing data.",
        "derivative": "A derivative measures the rate of change of a function. It represents the slope of a curve at any point. Applications: physics (velocity), economics (marginal cost), optimization problems.",
        "integral": "An integral is the reverse of derivative - it calculates the area under a curve. Used in physics (displacement), engineering (center of mass), and probability distributions.",
        "matrix": "A matrix is a rectangular array of numbers in rows and columns. Used in graphics transformations, machine learning, solving systems of equations, and neural networks.",
        "ai": "Artificial Intelligence enables computers to perform human-like tasks. Applications: virtual assistants, recommendations, autonomous vehicles, medical diagnosis.",
        "machine learning": "Machine Learning is AI that learns patterns from data without explicit programming. Types: supervised, unsupervised, reinforcement learning.",
    }
    
    for key, answer in fallback_answers.items():
        if key in question_lower:
            return answer
    
    # Ultimate fallback
    return f"""I understand you're asking about: {question}

Based on this question, here's a general approach to learning this concept:

1. **Start with Basics**: Understand the fundamental definition and core principles
2. **Visual Learning**: Use diagrams, graphs, or visual representations when possible
3. **Examples**: Study real-world applications and worked examples
4. **Practice**: Work through problems to solidify your understanding
5. **Connect Concepts**: Link this to related topics you already know

**Tips for better learning**:
- Ask specific questions about aspects you don't understand
- Break complex ideas into smaller parts
- Review periodically to strengthen memory
- Teach others to deepen your understanding

Feel free to ask follow-up questions for clarification!"""
    
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

        "what is vector": """A Vector is a mathematical quantity that has both magnitude and direction.

1. **Definition**:
   - A vector represents a quantity with size (magnitude) and direction
   - Examples: velocity, force, displacement, acceleration
   - Represented as an arrow with a specific length and direction

2. **Vector Representation**:
   - In 2D: v = (x, y) where x and y are components
   - In 3D: v = (x, y, z)
   - General: v = (v₁, v₂, ..., vₙ)

3. **Key Properties**:
   - **Magnitude**: Length of the vector = √(x² + y²)
   - **Direction**: Angle it makes with reference axis
   - **Components**: Projections on axes

4. **Vector Operations**:
   - **Addition**: v₁ + v₂ (head-to-tail method)
   - **Scalar Multiplication**: k·v (scales the vector)
   - **Dot Product**: v₁·v₂ = |v₁||v₂|cos(θ)
   - **Cross Product**: v₁ × v₂ (3D only, perpendicular vector)

5. **Vs Scalar**:
   - Scalar: Only magnitude (5 kg, 10°C)
   - Vector: Magnitude + direction (5 m/s north, 10 N upward)

6. **Real-World Applications**:
   - Physics: Forces, velocity, acceleration
   - Computer Graphics: 3D transformations
   - Machine Learning: Feature vectors representing data
   - Navigation: Displacement and direction

7. **Example**:
   - "5 units to the right and 3 units up" = vector (5, 3)
   - Magnitude = √(25 + 9) = √34 ≈ 5.83 units""",

        "what is derivative": """A Derivative measures how a function changes at any given point - the rate of change.

1. **Basic Definition**:
   - The derivative of f(x) at point x is the slope of the tangent line
   - Notation: f'(x), df/dx, or dy/dx
   - Represents instantaneous rate of change

2. **Geometric Interpretation**:
   - The slope of the curve at a specific point
   - Steeper slope = faster change
   - Positive slope = increasing, Negative slope = decreasing

3. **Common Derivatives**:
   - d/dx(xⁿ) = n·xⁿ⁻¹
   - d/dx(sin x) = cos x
   - d/dx(eˣ) = eˣ
   - d/dx(ln x) = 1/x

4. **Rules of Differentiation**:
   - Power Rule, Product Rule, Quotient Rule, Chain Rule

5. **Real-World Applications**:
   - Physics: Velocity (derivative of position)
   - Economics: Marginal cost and revenue
   - Engineering: Optimization problems
   - Medicine: Rate of drug concentration

6. **Physical Meaning**:
   - If s(t) = position, then s'(t) = velocity
   - If v(t) = velocity, then v'(t) = acceleration""",

        "what is integral": """An Integral is the reverse of a derivative - it finds the area under a curve and accumulates change.

1. **Basic Definition**:
   - The integral of f(x) gives the area under the curve
   - Notation: ∫f(x)dx
   - Two types: Definite and Indefinite

2. **Indefinite Integral (Antiderivative)**:
   - ∫f(x)dx = F(x) + C (where C is a constant)
   - Finds the general function whose derivative is f(x)

3. **Definite Integral**:
   - ∫ₐᵇ f(x)dx = Area under curve from x=a to x=b
   - Gives a specific numerical result

4. **Fundamental Theorem of Calculus**:
   - Connects derivatives and integrals
   - Integration and differentiation are inverse operations

5. **Common Integrals**:
   - ∫xⁿdx = xⁿ⁺¹/(n+1) + C
   - ∫sin x dx = -cos x + C
   - ∫eˣ dx = eˣ + C

6. **Applications**:
   - Physics: Finding displacement from velocity
   - Engineering: Center of mass calculations
   - Economics: Total revenue from marginal revenue
   - Probability: Computing probabilities from distributions""",

        "what is matrix": """A Matrix is a rectangular array of numbers organized in rows and columns.

1. **Definition**:
   - Array with m rows and n columns: m × n matrix
   - Example: 2×3 matrix has 2 rows and 3 columns
   - [1 2 3]
     [4 5 6]

2. **Notation**:
   - A, B, C... for matrix names
   - aᵢⱼ represents element in row i, column j
   - Matrix dimensions written as m × n

3. **Types of Matrices**:
   - Square Matrix: m = n
   - Row Matrix: 1 × n
   - Column Matrix: m × 1
   - Identity Matrix: Ones on diagonal, zeros elsewhere
   - Zero Matrix: All elements are zero

4. **Matrix Operations**:
   - **Addition**: Add corresponding elements
   - **Subtraction**: Subtract corresponding elements
   - **Multiplication**: A(m×n) × B(n×p) = C(m×p)
   - **Transpose**: Switch rows and columns

5. **Determinant and Inverse**:
   - Determinant: Special value for square matrices
   - Inverse: A⁻¹ such that A × A⁻¹ = I

6. **Applications**:
   - Computer Graphics: Transformations (rotation, scaling)
   - Machine Learning: Data representation, neural networks
   - Physics: System of equations
   - Engineering: Structural analysis
   - Cryptography: Encryption algorithms""",
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
async def auto_answer(doubt_id: str):
    for doubt in doubts_db:
        if doubt["id"] == doubt_id:
            # Call async function
            answer_content = await generate_ai_answer(doubt["question"])
            ai_answer = {
                "id": str(uuid.uuid4()),
                "doubt_id": doubt_id,
                "content": answer_content,
                "votes": 0,
                "is_ai": True
            }
            answers_db.append(ai_answer)
            return ai_answer
    return {"error": "Doubt not found"}
