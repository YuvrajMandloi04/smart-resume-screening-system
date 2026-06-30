# Smart Resume Screening System (AI-Powered)

A lightweight system that takes a Job Description (JD) and a Resume, and returns a relevance match score (0–100) with matched skills, missing skills, and a short explanation — built with FastAPI and TF-IDF + Cosine Similarity.

## Features

- Extracts candidate skill phrases (unigrams, bigrams, trigrams) from both JD and resume text
- Matches resume skills against JD requirements
- Computes a relevance score by blending skill overlap with TF-IDF cosine similarity
- Returns matched skills, missing skills, and a human-readable explanation
- Simple FastAPI backend with one POST endpoint
- Single-file HTML frontend to test the system visually, no build step required

## Tech Stack

- **Backend:** FastAPI, Python
- **Matching Logic:** scikit-learn (TfidfVectorizer, cosine_similarity)
- **Frontend:** Plain HTML, CSS, and JavaScript (single file, no framework)

## Project Structure

```
.
├── main.py            # FastAPI app and API route
├── matcher.py          # Core matching logic (skill extraction + TF-IDF scoring)
├── index.html           # Frontend UI to test the API
├── requirements.txt   # Python dependencies
└── README.md
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YuvrajMandloi04/smart-resume-screening-system.git
   cd smart-resume-screening-system
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the API

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Interactive API docs (Swagger UI) are available at `http://127.0.0.1:8000/docs`, where you can test the endpoint directly.

### Endpoint

**POST** `/match-resume`

**Request body:**
```json
{
  "jd_text": "We are looking for a Python Developer with experience in FastAPI, SQL, Machine Learning, and Data Science.",
  "resume_text": "Experienced Python developer skilled in FastAPI, Flask, and SQL. Worked on Machine Learning projects using scikit-learn and pandas."
}
```

**Response:**
```json
{
  "match_score": 78,
  "matched_skills": ["python", "fastapi", "sql", "machine learning"],
  "missing_skills": ["data science", "docker", "aws"],
  "explanation": "Strong match for this role. Matched 4 key terms including: fastapi, machine learning, python, sql. Missing important terms: aws, data science, docker."
}
```

## How to Run the Frontend

1. Make sure the backend is running (`uvicorn main:app --reload`).
2. Open `index.html` directly in your browser (double-click the file, or use the "Live Server" extension in VS Code).
3. Confirm the **API endpoint** field at the top of the page reads `http://127.0.0.1:8000/match-resume`.
4. Paste a job description into the left box and a resume into the right box.
5. Click **Run Match** to see the score, matched skills, missing skills, and explanation rendered on screen.

> Note: CORS is enabled in `main.py` so the browser-based frontend can call the API directly, even when opened as a local file.

## Approach

1. **Skill Extraction:** Text from both the JD and the resume is lowercased and split into segments using punctuation as boundaries. Each segment is broken into unigrams, bigrams, and trigrams (e.g. "python", "machine learning", "natural language processing"). A stopword list filters out common English words and generic recruiting terms (e.g. "experience", "team", "responsibilities") so only meaningful skill-like phrases remain.

2. **Skill Matching:** JD skill phrases are deduplicated to keep only the longest, most specific phrases (so "machine learning" is kept over a redundant overlapping "learning"). Each JD skill is checked against the resume's extracted skills using substring/phrase matching, producing a list of matched and missing skills.

3. **Similarity Scoring:** A TF-IDF vectorizer (using the same tokenization as skill extraction, with unigrams through trigrams) converts both texts into vectors, and cosine similarity is computed between them to capture overall textual closeness beyond just exact skill matches.

4. **Final Score:** The match score blends two signals — skill overlap ratio (70% weight) and TF-IDF cosine similarity (30% weight) — then scales the result to 0–100. Skill overlap is weighted higher since exact skill presence is generally a stronger signal of fit than general text similarity.

5. **Explanation Generation:** A short, templated explanation is generated based on the score band (strong / moderate / weak / poor match), along with a summary of which key terms were matched and which were missing.

## Example (via matcher.py directly)

```bash
python matcher.py
```

This runs a built-in test case and prints the match result as JSON to the console.

## Future Improvements

- Replace TF-IDF with sentence embeddings (e.g. Sentence-Transformers) for deeper semantic matching
- Add PDF/DOCX resume upload and parsing instead of plain text input
- Support batch scoring of multiple resumes against one JD, with ranking
- Add experience-level extraction (years of experience) as a secondary scoring factor
