import google.generativeai as genai

def gemini_evaluation_node(state: dict) -> dict:
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

# Example usage
if __name__ == "__main__":
    input_state = {
        "suggested_answer": "AI is the simulation of human intelligence by machines that perform tasks like learning and decision-making.",
        "student_answer": "AI is when machines can think and make decisions like humans."
    }

    updated_state = gemini_evaluation_node(input_state)
    print("Updated State:", updated_state)
