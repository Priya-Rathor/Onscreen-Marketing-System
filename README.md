# 🤖 Onscreen-Marketing-System AI

This is a powerful **FastAPI-based API** that evaluates student answers by comparing them with suggested answers using both **LLMs** (Gemini) and **embedding models** (SBERT, MiniLM, LaBSE). It returns an auto-generated score and feedback, useful for educational assessments, edtech platforms, or intelligent grading systems.

---

## 🚀 Features

- ✅ Uses **Gemini LLM** for scoring and one-line feedback
- ✅ Embedding-based similarity scoring with:
  - SBERT (`paraphrase-MiniLM-L6-v2`)
  - MiniLM (`paraphrase-MiniLM-L6-v2`)
  - LaBSE (`sentence-transformers/LaBSE`)
- ✅ Returns scores on a scale of 0–10 for each model
- ✅ FastAPI-powered endpoint for batch evaluation
- ✅ CORS enabled (useful for frontend testing)

---

## 📦 Tech Stack

- **Python 3.11+**
- **FastAPI** for backend
- **Google Generative AI (Gemini)** for LLM evaluation
- **SentenceTransformers** for embedding-based models
- **Sklearn** for cosine similarity
- **Uvicorn** for ASGI server

---

## 🧠 How It Works

Each student answer is compared to the suggested answer using the following:

| Model        | Method                          | Output           |
|--------------|----------------------------------|------------------|
| Gemini       | LLM prompt + response            | Score + Feedback |
| SBERT        | Embedding similarity             | Score (0–10)     |
| MiniLM       | Embedding similarity             | Score (0–10)     |
| LaBSE        | Embedding similarity             | Score (0–10)     |

---

## 📁 Project Structure

```

.
├── main.py                    # Main FastAPI app with all models and endpoints
├── requirements.txt           # Python dependencies
└── .env                       # API key (Gemini)

````

---

## 🧪 API Usage

### 🔗 POST `/evaluate`

Evaluate one or more student answers.

#### ✅ Request Body (JSON)
```json
[
  {
    "suggested_answer": "AI simulates human intelligence.",
    "student_answer": "AI makes machines think."
  }
]
````

#### ✅ Sample Response

```json
{
  "results": [
    {
      "suggested_answer": "AI simulates human intelligence.",
      "student_answer": "AI makes machines think.",
      "gemini_score": 8.5,
      "feedback": "Good answer but slightly lacks depth.",
      "sbert_score": 9.2,
      "minilm_score": 9.1,
      "labse_score": 8.8
    }
  ]
}
```

---

## ▶️ Running Locally

### 1️⃣ Clone the repo

```bash
git clone https://github.com/Priya-Rathor/student-evaluation-api.git
cd student-evaluation-api
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Create `.env` file

```env
GOOGLE_API_KEY=your-gemini-api-key
```

> 🔐 Or directly paste your Gemini API key in `genai.configure()` in `main.py`.

### 4️⃣ Run the app

```bash
uvicorn main:app --reload --port 7100
```

---

## 🌐 Test the Endpoint

You can test it using:

* 🔵 [Thunder Client](https://www.thunderclient.com/)
* 🟣 Postman
* 🔶 Curl
* 🌐 Your frontend via CORS

---

## 🛡️ Error Handling

* If LLM or model fails, it gracefully returns `0.0` as score.
* Error logs are printed to the console for debugging.

---

## 🧑‍💻 Author

**Priya Rathor**
🔗 [GitHub Profile](https://github.com/Priya-Rathor)

---

## 📌 To-Do (Optional)

* [ ] Add MongoDB to store evaluation reports
* [ ] Add retry logic for Gemini API
* [ ] Add test cases using `pytest`
* [ ] Add frontend form for demo

---

