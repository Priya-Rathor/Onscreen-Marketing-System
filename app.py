from fastapi import FastAPI, HTTPException
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from torch.nn.functional import cosine_similarity
import uvicorn
from langchain.prompts import PromptTemplate
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os
import openai  # Add OpenAI import

# Load environment variables from .env file
load_dotenv()

# Now your OpenAI API key will be available as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

class InputState(TypedDict):
    suggested_answer: str
    student_answer: str

class OutputState(TypedDict):
    bert_score: float
    roberta_score: float
    distilbert_score: float
    t5_score: float
    use_score: float
    gpt_score: float
    final_score: float

# Combine InputState and OutputState into OverallState
class OverallState(InputState, OutputState):
    pass

# Initialize FastAPI app
app = FastAPI()

# Global variables for models and tokenizers
models = {}
tokenizers = {}

# Initialize models
use_model = SentenceTransformer("all-mpnet-base-v2")

def initialize_models():
    """
    Initialize models and tokenizers.
    """
    global models, tokenizers

    logging.info("Initializing models...")

    # GPT-4 model
    models["gpt"] = ChatOpenAI(model="gpt-4", temperature=0)
    tokenizers["gpt"] = PromptTemplate(
        input_variables=["suggested_answer", "student_answer"],
        template="""Suggested answer: "{suggested_answer}" 
                    Student’s answer: "{student_answer}" 
                    Evaluate how well the student’s answer matches the suggested answer on a scale from 0 to 10, 
                    considering correctness, completeness, and clarity. Provide only a numeric score."""
    )

    # Transformer models
    model_configs = {
        "bert": "textattack/bert-base-uncased-MRPC",
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-uncased",
        "t5": "t5-small",
    }

    for model_name, model_path in model_configs.items():
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
        models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_path)

    logging.info("Models initialized successfully.")

def evaluate_with_transformer(model_name: str, suggested_answer: str, student_answer: str) -> float:
    """
    Evaluate similarity using a transformer model.
    """
    try:
        tokenizer = tokenizers[model_name]
        model = models[model_name]

        inputs = tokenizer(
            suggested_answer,
            student_answer,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        similarity_score = probabilities[0][1].item()
        return round(similarity_score * 10, 2)
    except Exception as e:
        logging.error(f"Error in {model_name} evaluation: {e}")
        return 0.0

def compute_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    try:
        vec1 = vec1.unsqueeze(0) if vec1.dim() == 1 else vec1
        vec2 = vec2.unsqueeze(0) if vec2.dim() == 1 else vec2
        similarity = cosine_similarity(vec1, vec2).item()
        return round(similarity, 2)
    except Exception as e:
        logging.error(f"Error computing cosine similarity: {e}")
        return 0.0

def use_evaluation(suggested_answer: str, student_answer: str) -> float:
    """
    Evaluate similarity using Universal Sentence Encoder.
    """
    try:
        embeddings = use_model.encode([suggested_answer, student_answer])
        return round(compute_cosine_similarity(torch.tensor(embeddings[0]), torch.tensor(embeddings[1])) * 10, 2)
    except Exception as e:
        logging.error(f"Error in USE evaluation: {e}")
        return 0.0

def gpt_evaluation(suggested_answer: str, student_answer: str) -> float:
    """
    Evaluate similarity using GPT-4.
    """
    try:
        prompt = tokenizers["gpt"].format(
            suggested_answer=suggested_answer,
            student_answer=student_answer
        )
        # Pass the prompt directly to GPT-4 using the langchain model
        response = models["gpt"](prompt)
        return round(float(response.strip()), 2)
    except Exception as e:
        logging.error(f"Error in GPT evaluation: {e}")
        return 0.0

@app.get("/")
async def root():
    return {"message": "Welcome to the Evaluation API"}

@app.post("/evaluate", response_model=OverallState)
async def evaluate_answers(request: InputState):
    """
    Evaluate the student's answer using multiple models and return scores.
    """
    suggested = request['suggested_answer'].strip()
    student = request['student_answer'].strip()

    try:
        bert_score = evaluate_with_transformer("bert", suggested, student)
        roberta_score = evaluate_with_transformer("roberta", suggested, student)
        distilbert_score = evaluate_with_transformer("distilbert", suggested, student)
        t5_score = evaluate_with_transformer("t5", suggested, student)
        use_score = use_evaluation(suggested, student)
        gpt_score = gpt_evaluation(suggested, student)

        # Compute final score as an average of all the individual model scores
        final_score = round((bert_score + roberta_score + distilbert_score + t5_score + use_score + gpt_score) / 6, 2)

        # Ensure all required fields are included in the response
        return OverallState(
            suggested_answer=suggested,
            student_answer=student,
            bert_score=bert_score,
            roberta_score=roberta_score,
            distilbert_score=distilbert_score,
            t5_score=t5_score,
            use_score=use_score,
            gpt_score=gpt_score, 
            final_score=final_score
        )
    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    initialize_models()
    uvicorn.run(app, host="127.0.0.1", port=7100)
