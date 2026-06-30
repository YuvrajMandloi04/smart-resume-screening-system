import re,json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# A basic stopword list (common English words that are not skills)
STOPWORDS = {
    "the", "is", "are", "was", "were", "be", "been", "being", "a", "an", "and",
    "or", "but", "if", "then", "else", "for", "to", "of", "in", "on", "at", "by",
    "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "than", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "we", "you", "your",
    "i", "he", "she", "it", "they", "them", "his", "her", "their", "our", "us",
    "this", "that", "these", "those", "as", "have", "has", "had", "do", "does",
    "did", "having", "would", "could", "must", "shall", "also", "etc", "using",
    "use", "work", "working", "experience", "years", "year", "looking", "required",
    "must", "preferred", "responsibilities", "requirements", "skills", "ability",
    "strong", "good", "knowledge", "team", "role", "job", "candidate", "company",
    "need", "needs", "needed", "familiarity", "familiar", "proficient", "proficiency",
    "demonstrated", "ideal", "plus", "bonus", "nice",
}

MIN_WORD_LENGTH = 2
MAX_NGRAM = 3  # support unigrams, bigrams, trigrams (e.g. "python", "machine learning", "natural language processing")


def _clean_tokens(text: str) -> list:
 
    text = text.lower()
    # Split on punctuation/newlines that mark a boundary between separate items
    raw_segments = re.split(r"[,;:\n\(\)\[\]/|]+|\.\s|\.$", text)

    segments = []
    for seg in raw_segments:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]*", seg)
        tokens = [tok.strip(".") for tok in tokens if len(tok.strip(".")) >= MIN_WORD_LENGTH]
        if tokens:
            segments.append(tokens)
    return segments


def extract_skills(text: str) -> set:
    
    segments = _clean_tokens(text)
    raw_skills = set()

    for tokens in segments:
        n = len(tokens)
        for size in range(1, MAX_NGRAM + 1):
            for i in range(n - size + 1):
                gram = tokens[i:i + size]

                # Reject if ANY word in the phrase is a stopword
                # (keeps "machine learning", drops "docker and aws" / "a python")
                if any(tok in STOPWORDS for tok in gram):
                    continue

                raw_skills.add(" ".join(gram))

    return raw_skills


def match_skills(jd_skills: set, resume_skills: set) -> tuple:
   
    # Only keep the longest JD phrases that aren't sub-phrases of a longer JD phrase
    longest_jd = sorted(jd_skills, key=lambda s: -len(s.split()))
    top_level_jd = []
    for phrase in longest_jd:
        if not any(phrase != longer and f" {phrase} " in f" {longer} " for longer in top_level_jd):
            top_level_jd.append(phrase)

    matched, missing = [], []
    for jd_phrase in top_level_jd:
        found = any(
            jd_phrase == r_phrase or f" {jd_phrase} " in f" {r_phrase} " or f" {r_phrase} " in f" {jd_phrase} "
            for r_phrase in resume_skills
        )
        (matched if found else missing).append(jd_phrase)

    return sorted(matched), sorted(missing)


def _tokenize_for_tfidf(text: str) -> list:
  
    segments = _clean_tokens(text)
    tokens = []
    for seg in segments:
        tokens.extend(seg)
    return tokens


def compute_similarity(jd_text: str, resume_text: str) -> float:
   
    vectorizer = TfidfVectorizer(tokenizer=_tokenize_for_tfidf, preprocessor=lambda x: x, ngram_range=(1, MAX_NGRAM))
    tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])  # type: ignore
    return float(similarity[0][0])


def generate_explanation(score: int, matched: list, missing: list) -> str:

    if score >= 75:
        verdict = "Strong match for this role."
    elif score >= 50:
        verdict = "Moderate match — candidate covers a good portion of requirements."
    elif score >= 25:
        verdict = "Weak match — significant gaps in required skills."
    else:
        verdict = "Poor match — resume does not align well with this JD."

    matched_part = f"Matched {len(matched)} key terms including: {', '.join(matched[:5])}." if matched else "No strong matching terms found."
    missing_part = f"Missing important terms: {', '.join(missing[:5])}." if missing else "No major gaps detected."

    return f"{verdict} {matched_part} {missing_part}"


def match_resume_to_jd(jd_text: str, resume_text: str) -> dict:
    
    # Step 1: Extract candidate skills from both
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)

    matched_skills, missing_skills = match_skills(jd_skills, resume_skills)

    # Step 2: TF-IDF cosine similarity (0 to 1)
    similarity = compute_similarity(jd_text, resume_text)

    # Step 3: Skill overlap ratio (0 to 1) - what fraction of JD's top-level
    # skills were found in the resume
    total_jd_skills = len(matched_skills) + len(missing_skills)
    overlap_ratio = (len(matched_skills) / total_jd_skills) if total_jd_skills > 0 else 0.0

    # Step 4: Blend the two signals (skill overlap weighted higher - 70/30)
    blended = (0.7 * overlap_ratio) + (0.3 * similarity)
    score = round(blended * 100)

    # Step 5: Explanation
    explanation = generate_explanation(score, matched_skills, missing_skills)

    return {
        "match_score": score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "explanation": explanation,
    }


# Quick manual test when running this file directly
if __name__ == "__main__":
    jd = """
    We are looking for a Python Developer with experience in FastAPI, SQL,
    Machine Learning, and Data Science. Knowledge of Docker and AWS is a plus.
    """

    resume = """
    Experienced Python developer skilled in FastAPI, Flask, and SQL.
    Worked on Machine Learning projects using scikit-learn and pandas.
    Familiar with Git and basic Docker usage.
    """

    result = match_resume_to_jd(jd, resume)
    print(json.dumps(result, indent=2))