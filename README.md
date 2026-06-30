# Smart Resume Screening System

AI-powered resume vs Job Description matcher using TF-IDF + Cosine Similarity.

## Files
- `matcher.py` — Core logic (skill extraction + TF-IDF matching + explanation)
- `main.py` — FastAPI app exposing the `/match-resume` endpoint
- `requirements.txt` — Python dependencies

## How to run in VS Code

### 1. Open the folder
Open the `resume_screener` folder in VS Code (File → Open Folder).

### 2. Create a virtual environment (recommended)
Open VS Code's terminal (Terminal → New Terminal) and run:

```bash
python -m venv venv
```

Activate it:
- **Windows**: `venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
uvicorn main:app --reload
```

You should see something like:
```
Uvicorn running on http://127.0.0.1:8000
```

### 5. Test the API
Open your browser and go to:
```
http://127.0.0.1:8000/docs
```

This opens FastAPI's interactive Swagger UI. Click on `POST /match-resume` → "Try it out" → paste your JD and resume text → Execute.

### Example request body
```json
{
  "jd_text": "Looking for Python Developer with FastAPI, SQL, Machine Learning experience.",
  "resume_text": "Python developer skilled in FastAPI, Flask, SQL, and Machine Learning projects."
}
```

### Example response
```json
{
  "match_score": 55,
  "matched_skills": ["developer", "fastapi", "learning", "machine", "python", "sql"],
  "missing_skills": [],
  "explanation": "Moderate match — candidate covers a good portion of requirements. Matched 6 key terms including: developer, fastapi, learning, machine, python. No major gaps detected."
}
```

### Quick test via curl (optional, instead of Swagger UI)
```bash
curl -X POST http://127.0.0.1:8000/match-resume -H "Content-Type: application/json" -d "{\"jd_text\": \"Python Developer with FastAPI, SQL\", \"resume_text\": \"Python, FastAPI, SQL developer\"}"
```

## How it works (quick recap)
1. **Skill extraction**: Both JD and resume text are lowercased, split into word tokens, and stopwords are removed. Whatever remains is treated as a "skill/keyword".
2. **Matching**: 
   - Matched skills = words common to both JD and resume
   - Missing skills = words in JD but not in resume
3. **Score**: TF-IDF vectors are built for JD and resume, then Cosine Similarity between them is calculated and scaled to 0–100.
4. **Explanation**: A short verdict (Strong/Moderate/Weak/Poor match) plus a summary of matched/missing terms.
