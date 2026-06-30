from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from matcher import match_resume_to_jd

app = FastAPI(
    title="Smart Resume Screening System",
    description="AI-powered resume vs JD matching using TF-IDF + Cosine Similarity",
    version="1.0.0",
)

# Allow the frontend (index.html, opened from file:// or any localhost port)
# to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    jd_text: str = Field(..., description="Job Description text")
    resume_text: str = Field(..., description="Resume text (plain text)")


class MatchResponse(BaseModel):
    match_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    explanation: str


@app.get("/")
def root():
    return {"message": "Smart Resume Screening System is running. Go to /docs to try the API."}


@app.post("/match-resume", response_model=MatchResponse)
def match_resume(payload: MatchRequest):
    result = match_resume_to_jd(payload.jd_text, payload.resume_text)
    return result

@app.post("/match-resume", response_model=MatchResponse)
def match_resume(payload: MatchRequest):
    result = match_resume_to_jd(payload.jd_text, payload.resume_text)
    return result
