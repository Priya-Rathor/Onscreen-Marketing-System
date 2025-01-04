from fastapi import FastAPI, HTTPException
import torch
import logging
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from torch.nn.functional import cosine_similarity
import uvicorn
from langchain.prompts import PromptTemplate
from typing import Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class InputState(TypedDict):
    suggested_answer: str
    student_answer: str

class OutputState(TypedDict):
    student_answer: str
    sbert_score: float
    roberta_score: Optional[float]
    distilroberta_score: Optional[float]
    t5_score: Optional[float]
    use_score: float
    gpt_score: float
    minilm_score: Optional[float]
    electra_score: Optional[float]
    labse_score: Optional[float]
    feedback: str

# Initialize FastAPI app
app = FastAPI()

# Global variables for models
models = {}

def initialize_models():
    """
    Initialize models and tokenizers.
    """
    global models
    logging.info("Initializing models...")
    models["sbert"] = SentenceTransformer("all-MiniLM-L6-v2")
    models["use"] = SentenceTransformer("all-mpnet-base-v2")
    models["gpt"] = ChatOpenAI(model="gpt-4", temperature=0)
    logging.info("Models initialized successfully.")

def compute_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    vec1 = vec1.unsqueeze(0) if vec1.dim() == 1 else vec1
    vec2 = vec2.unsqueeze(0) if vec2.dim() == 1 else vec2
    similarity = cosine_similarity(vec1, vec2).item()
    return round(similarity, 2)

def evaluate_with_transformer(model_name: str, suggested_answer: str, student_answer: str) -> float:
    """
    Evaluate similarity using a transformer model.
    """
    try:
        model = models[model_name]
        if model_name == "gpt":
            prompt = f"""Suggested answer: "{suggested_answer}" 
Student’s answer: "{student_answer}" 
Evaluate how well the student’s answer matches the suggested answer on a scale from 0 to 10."""
            response = model(prompt)
            return round(float(response.strip()), 2)
        else:
            embeddings = model.encode([suggested_answer, student_answer])
            similarity_score = compute_cosine_similarity(
                torch.tensor(embeddings[0]), torch.tensor(embeddings[1])
            )
            return round(similarity_score * 10, 2)
    except Exception as e:
        logging.error(f"Error in {model_name} evaluation: {e}")
        return 0.0

@app.get("/")
async def root():
    return {"message": "Welcome to the Evaluation API"}

@app.post("/evaluate", response_model=OutputState)
async def evaluate_answers(request: InputState):
    """
    Evaluate the student's answer using multiple models and return scores.
    """
    suggested = request["suggested_answer"].strip()
    student = request["student_answer"].strip()

    try:
        sbert_score = evaluate_with_transformer("sbert", suggested, student)
        use_score = evaluate_with_transformer("use", suggested, student)
        gpt_score = evaluate_with_transformer("gpt", suggested, student)

        # Placeholder values for other scores
        roberta_score = None
        distilroberta_score = None
        t5_score = None
        minilm_score = None
        electra_score = None
        labse_score = None

        feedback = "Good answer."  # Example feedback

        return OutputState(
            student_answer=student,
            sbert_score=sbert_score,
            roberta_score=roberta_score,
            distilroberta_score=distilroberta_score,
            t5_score=t5_score,
            use_score=use_score,
            gpt_score=gpt_score,
            minilm_score=minilm_score,
            electra_score=electra_score,
            labse_score=labse_score,
            feedback=feedback,
        )
    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    initialize_models()
    uvicorn.run(app, host="127.0.0.1", port=7100)
