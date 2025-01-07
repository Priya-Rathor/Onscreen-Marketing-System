import os
import openai
from typing import TypedDict
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import END, START, StateGraph
import torch
from transformers import ElectraForPreTraining ,ElectraTokenizerFast
from typing import List


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
    roberta_score:float
    distilroberta_score:float
    t5_score:float
    minilm_score:float
    electra_score:float
    labse_score:float


app = FastAPI()

#---------------------------------------------------------------------- 
#                       Open AI model 
#----------------------------------------------------------------------
def llm_evaluation_node(state: InputState) -> OutputState:
    """Use OpenAI API to evaluate the student answer and return llm_score with feedback."""
    
    prompt = f"""
    Suggested answer: "{state['suggested_answer']}"
    Student’s answer: "{state['student_answer']}"
    
    Evaluate how well the student’s answer matches the suggested answer on a scale from 0 to 10,
    considering correctness, completeness, and clarity. Provide a numeric score followed by a one-line feedback.
    Example: "8.5 - Good answer but slightly lacks depth."
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an AI grader that evaluates student answers."},
                      {"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0
        )
        output = response["choices"][0]["message"]["content"].strip()

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
    state["llm_score"] = score
    state["llm_feedback"] = feedback
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
#                      Roberta Model
#---------------------------------------------------------------- 
roberta_model = SentenceTransformer('all-roberta-large-v1')

def roberta_evaluation_node(state: InputState) -> OutputState:
    """
    Use a RoBERTa-based model (via SentenceTransformer) to calculate
    the cosine similarity between suggested_answer and student_answer,
    and scale the result from 0–10.

    :param state: A dictionary with keys 'suggested_answer' and 'student_answer'.
    :return: The updated dictionary with a new key 'roberta_score'.
    """
    try:
        # 1. Extract the two answers from the state
        sentences = [state['suggested_answer'], state['student_answer']]
        
        # 2. Encode both answers using the RoBERTa model
        embeddings = roberta_model.encode(sentences)
        
        # 3. Compute cosine similarity (range: 0 to 1)
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # 4. Scale similarity to a 0–10 range
        scaled_score = similarity * 10
        
        print(f"RoBERTa Similarity Score: {scaled_score:.2f}")
    except Exception as e:
        print(f"Exception in RoBERTa evaluation: {e}")
        scaled_score = 0.0
    
    # 5. Update and return the state with the new score
    state["roberta_score"] = scaled_score
    return state


#---------------------------------------------------------------- 
#                     Distiilroberta Model
#----------------------------------------------------------------
distilroberta_model = SentenceTransformer('all-distilroberta-v1')

def distilroberta_evaluation_node(state:InputState)->OutputState:
    """
    Use a DistilRoBERTa-based model (via SentenceTransformer) to calculate
    the cosine similarity between suggested_answer and student_answer,
    and scale the result from 0–10.

    :param state: A dictionary with keys 'suggested_answer' and 'student_answer'.
    :return: The updated dictionary with a new key 'distilroberta_score'.
    """

    try:
        sentences= [state['student_answer'],state['suggested_answer']]
        embeddings = distilroberta_model.encode(sentences)
        similarity = cosine_similarity([embeddings[0]],[embeddings[1]])[0][0]

        scaled_score = similarity*10

    except Exception as e:
        scaled_score=0.0

    state["distilroberta_score"] = scaled_score
    return state      


#---------------------------------------------------------------- 
#                      t5 model 
#----------------------------------------------------------------
t5_model = SentenceTransformer("sentence-t5-base")

def t5_evaluation_node(state:InputState)-> OutputState:
    """
    Use a T5-based model ti calculate the cosine similarity between suggested_answer and student_answer, 
    then scale the result from 0-10.
    """

    try:
        sentences =[state["suggested_answer"],state["student_answer"]]

        embeddings = t5_model.encode(sentences)

        similarity = cosine_similarity([embeddings[0]],[embeddings[1]])[0][0]

        scaled_score = similarity *10
      
    

    except Exception as e:   
        scaled_score:0.0

    state["t5_score"]= scaled_score
    return state    



#---------------------------------------------------------------- 
#                      USE Model
#---------------------------------------------------------------- 

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
#                      electra Model
#---------------------------------------------------------------- 

# discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator", force_download=True)
# tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator", force_download=True)
# print("Electra model loaded successfully.")

# def electra_evaluation_node(state:InputState)->OutputState:
#     """
#     Use the ELECTRA-base discriminator to compute embeddings for two texts,
#     calculate cosine similarity, and store a 0–10 scaled score in `state["electra_score"]`.

#     Expected keys in `state`:
#       - 'suggested_answer': str
#       - 'student_answer': str

#     Returns:
#       - The same state dict, updated with 'electra_score' (float).
#     """
#     try:
#         # 2. Prepare inputs
#         suggested_answer = state["suggested_answer"]
#         student_answer   = state["student_answer"]

#         # Tokenize each sentence separately
#         inputs_suggested = tokenizer(suggested_answer, return_tensors="pt", truncation=True, padding=True)
#         inputs_student   = tokenizer(student_answer, return_tensors="pt", truncation=True, padding=True)

#         # 3. Forward pass to get hidden states (disable gradient calculations)
#         with torch.no_grad():
#             outputs_suggested = discriminator(**inputs_suggested, output_hidden_states=True)
#             outputs_student   = discriminator(**inputs_student,   output_hidden_states=True)

#         # 4. Extract the last hidden state (shape: [batch_size, seq_len, hidden_dim])
#         last_hidden_suggested = outputs_suggested.hidden_states[-1]  # Tensor
#         last_hidden_student   = outputs_student.hidden_states[-1]

#         # 5. Apply mean pooling across the sequence dimension (dim=1) -> shape: [batch_size, hidden_dim]
#         embedding_suggested = last_hidden_suggested.mean(dim=1)
#         embedding_student   = last_hidden_student.mean(dim=1)

#         # 6. Compute cosine similarity (PyTorch functional)
#         #    shape: [batch_size], but we only have batch_size=1
#         cos_similarity = torch.nn.functional.cosine_similarity(embedding_suggested, embedding_student).item()

#         # 7. Scale the similarity to a 0–10 score
#         scale_score = cos_similarity * 10
#     except Exception as e:
#         print(f"Error during ELECTRA evaluation: {e}")
#         scale_score = 0.0

#     # 8. Update state and return
#     state["electra_score"] = scale_score
#     return state

#---------------------------------------------------------------- 
#                      labse Model
#---------------------------------------------------------------- 

labse_model = SentenceTransformer("sentence-transformers/LaBSE")

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
#                      Lang Graph 
#----------------------------------------------------------------
builder = StateGraph(InputState, OutputState)

builder.add_node("llm_node", llm_evaluation_node)
builder.add_node("sbert_node", sbert_evaluation_node)
builder.add_node("roberta_node",roberta_evaluation_node)
builder.add_node("distilroberta_node",distilroberta_evaluation_node)
builder.add_node("t5_node", t5_evaluation_node)
builder.add_node("minilm_node",minilm_evaluation_node)
#builder.add_node("electra_node",electra_evaluation_node)
builder.add_node("labse_node",labse_evaluation_node)

# Define flow of the graph
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", "sbert_node")
builder.add_edge("sbert_node", "roberta_node")
builder.add_edge("roberta_node","distilroberta_node")
builder.add_edge("distilroberta_node","t5_node")
builder.add_edge("t5_node","minilm_node")
builder.add_edge("minilm_node","labse_node")
#builder.add_edge("electra_node","labse_node")
builder.add_edge("labse_node",END)
graph = builder.compile()


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
    print("hello")

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
            # Directly invoke the evaluation nodes without graph.invoke
            state = llm_evaluation_node(state)  # Update state with LLM evaluation
            state = sbert_evaluation_node(state)  # Add SBERT evaluation results
            state = roberta_evaluation_node(state)  # Add RoBERTa evaluation results
            state = distilroberta_evaluation_node(state)  # Add DistilRoBERTa evaluation results
            state = t5_evaluation_node(state)  # Add T5 evaluation results
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
                    "llm_score": 0.0,
                    "sbert_score": 0.0,
                    "roberta_score": 0.0,
                    "distilroberta_score": 0.0,
                    "t5_score": 0.0,
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
    uvicorn.run(app, host="127.0.0.1", port=7100)
    
