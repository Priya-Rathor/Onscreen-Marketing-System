from typing import TypedDict
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

app =  FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define Input and Output State
class InputState(TypedDict):
    suggested_answer: str
    student_answer: str

class OutputState(TypedDict):
    gemini_score: float
    feedback: str
    sbert_score: float
    minilm_score:float
    labse_score:float


#---------------------------------------------------------------- 
#                       Gemini Model
#----------------------------------------------------------------


def gemini_evaluation_node(state: InputState) -> OutputState:
    """Use Gimmia Generative AI to evaluate the student answer and return llm_score with feedback."""

    # Configure Gimmia Generative AI
    genai.configure(api_key="AIzaSyDUiT3yPTTo2nmoPRj-hpo2r2OyrPH5cqs")
    model = genai.GenerativeModel("gemini-2.0-flash-exp")

    # Prepare the evaluation prompt
    prompt = f"""
    Suggested answer: "{state['suggested_answer']}"
    Student’s answer: "{state['student_answer']}"

    Evaluate how well the student’s answer matches the suggested answer on a scale from 0 to 10,
    considering correctness, completeness, and clarity. Provide a numeric score followed by a one-line feedback.
    Example: "8.5 - Good answer but slightly lacks depth."
    """

    try:
        # Generate content using the model
        response = model.generate_content(prompt)
        output = response.text.strip()

        if " - " in output:
            score, feedback = output.split(" - ", 1)
            score = float(score)
        else:
            score = 0.0
            feedback = "No valid feedback provided."

        print(f"LLM Evaluation Result: score={score}, feedback={feedback}")
    except Exception as e:
        print(f"Exception in LLM evaluation: {e}")
        score = 0.0
        feedback = "Error in LLM evaluation."

    # Update and return the state
    state["gemini_score"] = score
    state["feedback"] = feedback
    return state


#---------------------------------------------------------------- 
#                       SBERT Model 
#----------------------------------------------------------------
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def sbert_evaluation_node(state: InputState) -> OutputState:
    """Use SBERT to calculate the cosine similarity between suggested_answer and student_answer."""
    try:
        sentences = [state['suggested_answer'], state['student_answer']]
        embeddings = sbert_model.encode(sentences)
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        scaled_score = similarity * 10  # Scale similarity to 0-10
        
        print(f"SBERT Similarity Score: {scaled_score}")
    except Exception as e:
        print(f"Exception in SBERT evaluation: {e}")
        scaled_score = 0.0
    
    # Update and return the state
    state["sbert_score"] = scaled_score
    return state

#---------------------------------------------------------------- 
#                      Minilm model
#---------------------------------------------------------------- 

minilm_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def minilm_evaluation_node(state:InputState)->OutputState:
      """
    Use a MiniLM-based model (via SentenceTransformer) to calculate
    the cosine similarity between suggested_answer and student_answer,
    then scale the result from 0–10.
    
    :param state: A dictionary with keys 'suggested_answer' and 'student_answer'.
    :return: The updated dictionary with a new key 'minilm_score'.
    """
      try:
          
          sentences = [state["suggested_answer"],state["student_answer"]]

          embeddings = minilm_model.encode(sentences)

          similarity = cosine_similarity([embeddings[0]],[embeddings[1]])[0][0]

          scale_score = similarity*10

      except Exception as e:
          scale_score = 0.0

      state["minilm_score"]=scale_score
      return state    

#---------------------------------------------------------------- 
#                      labse Model
#---------------------------------------------------------------- 


labse_model = SentenceTransformer("sentence-transformers/LaBSE")

def labse_evaluation_node(state: InputState) -> OutputState:
    """
    Use a LaBSE model (via SentenceTransformer) to calculate
    the cosine similarity between 'suggested_answer' and 'student_answer',
    then scale the result from 0–10.

    :param state: A dictionary containing 'suggested_answer' and 'student_answer'.
    :return: The updated dictionary with a new key 'labse_score'.
    """
    scaled_score = 0.0  # Default value in case of exception
    try:
        sentences = [state["suggested_answer"], state["student_answer"]]

        # Generate embeddings for the two sentences
        embeddings = labse_model.encode(sentences)

        # Calculate cosine similarity between the embeddings
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # Scale the similarity to a score out of 10
        scaled_score = similarity * 10

    except Exception as e:
        # Log the exception if needed (e.g., using logging module)
        print(f"Error in LaBSE evaluation node: {e}")

    # Update the state with the score
    state["labse_score"] = scaled_score
    return state


#---------------------------------------------------------------- 
#                   FastAPI BaseModel
#----------------------------------------------------------------
class Item(BaseModel):
    suggested_answer: str
    student_answer: str
    
#---------------------------------------------------------------------- 
#                     FastAPI endpoint for Hello 
#----------------------------------------------------------------------

@app.post("/hello")
async def Hello():
    return ("hello")

#---------------------------------------------------------------------- 
#                     FastAPI endpoint for Evaluation
#----------------------------------------------------------------------
@app.post("/evaluate")
async def evaluate_items(items: List[Item]):
    states = []  # Initialize a list to store results

    for item in items:
        state = {
            "suggested_answer": item.suggested_answer,
            "student_answer": item.student_answer
        }

        try:
            state =gemini_evaluation_node(state)
            state = sbert_evaluation_node(state)  # Add SBERT evaluation results
            state = minilm_evaluation_node(state)  # Add MiniLM evaluation results
            state = labse_evaluation_node(state)  # Add LaBSE evaluation results

            # Append the fully processed state to the results list
            states.append(state)
        except Exception as e:
            print(f"Exception during evaluation for item: {e}")
            # Append a default error result for the failed evaluation
            states.append({
                "suggested_answer": item.suggested_answer,
                "student_answer": item.student_answer,
                "error": "Evaluation failed.",
                "scores": {
                    "gemini_score": 0.0,
                    "sbert_score": 0.0,
                    "minilm_score": 0.0,
                    "labse_score": 0.0
                }
            })

    # Return the results list as the API response
    return {"results": states}

#---------------------------------------------------------------- 
#                       Run FastAPI
#----------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7100)
    
