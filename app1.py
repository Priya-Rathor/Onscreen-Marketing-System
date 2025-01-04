import os
import openai
from typing import TypedDict
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import END, START, StateGraph, MessagesState

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define Input and Output State
class InputState(TypedDict):
    suggested_answer: str
    student_answer: str

class OutputState(TypedDict):
    llm_score: float
    llm_feedback: str
    sbert_score: float


app = FastAPI()

#---------------------------------------------------------------------- 
# Open AI model 
#----------------------------------------------------------------------

def llm_evaluation_node(state: InputState) -> OutputState:
    """Use OpenAI API to evaluate student answer and return llm_score with feedback."""
    
    prompt = f"""
    Suggested answer: "{state['suggested_answer']}"
    Student’s answer: "{state['student_answer']}"
    
    Evaluate how well the student’s answer matches the suggested answer on a scale from 0 to 10,
    considering correctness, completeness, and clarity. Provide a numeric score followed by a one-line feedback.
    Example: "8.5 - Good answer but slightly lacks depth."
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use GPT-4 or GPT-3.5-turbo
            messages=[{"role": "system", "content": "You are an AI grader that evaluates student answers."},
                      {"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )
        output = response["choices"][0]["message"]["content"].strip()

        # Split the response into score and feedback
        score, feedback = output.split(" - ", 1)
        score = float(score)
        
        # Debug print statement to check the state
        print(f"LLM Evaluation Result: score={score}, feedback={feedback}")
        
    except ValueError:
        score = 0.0
        feedback = "No valid feedback."
    except Exception as e:
        raise RuntimeError(f"Error during LLM evaluation: {e}")
    
    return {"llm_score": score, "llm_feedback": feedback}

#---------------------------------------------------------------- 
# SBERT Model 
#----------------------------------------------------------------
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def sbert_evaluation_node(state: InputState) -> OutputState:
    """Use SBERT to calculate the cosine similarity between suggested_answer and student_answer."""
    
    # Encode the sentences using SBERT
    sentences = [state['suggested_answer'], state['student_answer']]
    embeddings = sbert_model.encode(sentences)
    
    # Calculate cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Return the result as a dictionary
    return {
        "sbert_score": similarity * 10  # The SBERT score
    }

#---------------------------------------------------------------- 
# Lang Graph 
#----------------------------------------------------------------
builder = StateGraph(InputState, OutputState)

builder.add_node("llm_node", llm_evaluation_node)
builder.add_node("sbert_node", sbert_evaluation_node)

builder.add_edge(START, "llm_node")
builder.add_edge(START, "sbert_node")
builder.add_edge("llm_node",END)
builder.add_edge("sbert_node", END)
graph = builder.compile()

#---------------------------------------------------------------- 
# FastAPI BaseModel
#----------------------------------------------------------------

class Item(BaseModel):
    suggested_answer: str
    student_answer: str


#---------------------------------------------------------------------- 
# FastAPI endpoint for Evaluation
#----------------------------------------------------------------------

@app.post("/evaluate")
async def evaluate_item(item: Item):
    state = {
        "suggested_answer": item.suggested_answer,
        "student_answer": item.student_answer
    }
    result = graph.invoke(state)
    print(f"Graph result: {result}")
    
    llm_score = result.get('llm_score', None)
    llm_feedback = result.get('llm_feedback', None)
    sbert_score = result.get('sbert_score', None)

    # Combine the results into a single response
    return {
        "llm_score": llm_score,
        "llm_feedback": llm_feedback,
        "sbert_score": sbert_score
    }
#---------------------------------------------------------------- 
# Run FastAPI
#----------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7100)
