from fastapi import FastAPI
from pydantic import BaseModel, Field

from matcher import match_resume_to_jd

app = FastAPI(
    title="Smart Resume Screening System",
    description="AI-powered resume vs JD matching using TF-IDF + Cosine Similarity",
    version="1.0.0",
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