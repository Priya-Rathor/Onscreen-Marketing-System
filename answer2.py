import streamlit as st
import os
import warnings
from langchain_community.utilities import SerpAPIWrapper
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai

# Suppress warnings and logs
warnings.filterwarnings("ignore")
os.environ["SERPAPI_API_KEY"] = "4a188fc3b0edbd0bd61a8d54e7754300feeea8d765b6b853e5d1f3d76facc586"  # Replace with your SerpAPI key
genai.configure(api_key="AIzaSyBP-b2yWydI5T5IfHzebQQrrBnBXFcpAYA")  # Replace with your Google Generative AI key
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize utilities
search = SerpAPIWrapper()
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "questions" not in st.session_state:
    st.session_state.questions = []
if "student_answers" not in st.session_state:
    st.session_state.student_answers = []

# Function to evaluate answers
def evaluate_answer_with_fallback(question, user_answer, educator_answer=None, subject=None):
    

        
    comparison_basis = f"Educator's Expected Answer: {educator_answer}"
            
    
    search_result = search.run(question)
            
            
        # Generate Evaluation
    prompt = f"""
        Question: {question}
        educator answer {comparison_basis}
        serpapi :{search_result}
        User's Answer: {user_answer}
        Subject: {subject if subject else "General"}

        You are an AI evaluator tasked with assessing the user's answer to the given question. Evaluate the response based on the following criteria:
        1. **Accuracy**: How well does the user's answer align with the correct answer ?
        2. **Relevance**: Is the user's answer directly related to the question?
        3. **Completeness**: Does the user's answer include all key points or provide enough detail?
        4. **Clarity**: Is the user's answer well-explained and easy to understand?

        Provide a **score between 0 and 100**:
        - **100**: A completely accurate, relevant, complete, and clear response.
        - **0**: A completely incorrect, irrelevant, or nonsensical response.
        - Provide partial credit for answers that are partially correct, incomplete, or partially clear.

        Include detailed feedback:
        - Highlight **factual errors, omissions, or misunderstandings** in the user's answer.
        - Mistakes made by the user.
        - Solution: Provide the correct answer 

        Format the response as:
        1. **Score**: (numerical score out of 100)
        2. **Feedback**: (a concise explanation of the score, including strengths and weaknesses of the answer)
        3. **Mistakes**: (What mistakes were made by the user?)
        4. **Solution**: (The correct answer based on the .)
        """
    response = model.generate_content(prompt)
    return response.text.strip()
    
       

# Streamlit App
st.title("AI-Powered Q&A Evaluator for Educators and Students")

# User Role Selection
role = st.sidebar.selectbox("Select Your Role:", ["Educator", "Student"])

if role == "Educator":
    st.header("Educator Dashboard")
    st.write("Manage questions and evaluate student answers.")

    # Input for question creation
    question = st.text_input("Enter a question:", placeholder="Type your question here...")
    subject = st.text_input("Enter the subject (optional):", placeholder="e.g. Math, History")
    educator_answer = st.text_area("Enter the correct answer (optional):", placeholder="Provide your expected answer here...")

    if st.button("Save Question"):
        if question.strip():
            # Save question
            st.session_state.questions.append({
                "question": question,
                "subject": subject,
                "educator_answer": educator_answer.strip() if educator_answer else None,
            })
            st.success("Question saved successfully!")
        else:
            st.error("Please enter a valid question.")

    # View and evaluate student answers
    if st.button("Evaluate Student Answers"):
        if st.session_state.student_answers:
            st.write("### Student Answers for Evaluation:")
            for idx, answer in enumerate(st.session_state.student_answers):
                st.write(f"**Question:** {answer['question']} (Subject: {answer['subject']})")
                st.write(f"**Student's Answer:** {answer['answer']}")

                # AI Evaluation
                question_data = next((q for q in st.session_state.questions if q['question'] == answer['question']), {})
                if "evaluation" not in answer or not answer["evaluation"]:
                    st.write("Evaluating with AI...")
                    ai_evaluation = evaluate_answer_with_fallback(
                        answer['question'],
                        answer['answer'],
                        question_data.get("educator_answer"),
                        answer['subject']
                    )
                    answer["evaluation"] = ai_evaluation
                st.write("### AI Evaluation:")
                st.write(answer["evaluation"])

                # Input for educator's review
                feedback = st.text_area(f"Educator Feedback for Answer {idx + 1}:", answer["evaluation"], key=f"feedback_{idx}")
                if st.button(f"Finalize Evaluation for Answer {idx + 1}", key=f"finalize_{idx}"):
                    answer["evaluation"] = feedback
                    st.success(f"Evaluation for Answer {idx + 1} saved!")
        else:
            st.write("No student answers submitted yet.")

elif role == "Student":
    st.header("Student Dashboard")
    st.write("Answer questions and receive AI-generated evaluations, with final feedback from your educator.")

    # Select a question to answer
    if st.session_state.questions:
        question_idx = st.selectbox("Select a Question:", range(len(st.session_state.questions)))
        question_data = st.session_state.questions[question_idx]
        st.write(f"**Question:** {question_data['question']} (Subject: {question_data['subject']})")

        # Answer input
        user_answer = st.text_area("Your Answer:", placeholder="Type your answer here...")

        if st.button("Submit Answer"):
            if user_answer.strip():
                # Save the student's answer
                st.session_state.student_answers.append({
                    "question": question_data["question"],
                    "subject": question_data["subject"],
                    "answer": user_answer,
                    "evaluation": None,  # Placeholder for AI evaluation and educator feedback
                })
                st.success("Answer submitted successfully!")
            else:
                st.error("Please provide an answer before submitting.")
    else:
        st.write("No questions available. Please wait for your educator to create them.")

    # View evaluated answers
    if st.button("View Evaluations"):
        if st.session_state.student_answers:
            st.write("### Your Evaluated Answers:")
            for answer in st.session_state.student_answers:
                st.write(f"**Question:** {answer['question']} (Subject: {answer['subject']})")
                st.write(f"**Your Answer:** {answer['answer']}")
                st.write(f"**Evaluation:** {answer['evaluation'] if answer['evaluation'] else 'Evaluation pending.'}")
        else:
            st.write("No evaluations available yet.")

# Clear memory
if st.sidebar.button("Clear All Memory"):
    st.session_state.memory.clear()
    st.session_state.questions = []
    st.session_state.student_answers = []
    st.sidebar.success("All memory cleared.")
