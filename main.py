from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)

app = FastAPI(max_request_size=50 * 1024 * 1024)

# -------------------- SCHEMA --------------------

class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: str


class Case(BaseModel):
    case_id: str
    patient_id: str
    patient_name: str
    current_study: Study
    prior_studies: List[Study]


class RequestBody(BaseModel):
    challenge_id: str
    schema_version: int
    generated_at: str
    cases: List[Case]


# -------------------- HELPERS --------------------

def years_between(current_date, prior_date):
    try:
        c = datetime.strptime(current_date, "%Y-%m-%d")
        p = datetime.strptime(prior_date, "%Y-%m-%d")
        return abs((c - p).days) / 365.25
    except Exception:
        return 999


def normalize(text):
    return text.lower().replace("cntrst", "contrast")


def compute_similarities(current_text, prior_texts):
    vectorizer = TfidfVectorizer()
    corpus = [current_text] + prior_texts
    vectors = vectorizer.fit_transform(corpus)

    current_vec = vectors[0:1]
    prior_vecs = vectors[1:]

    return cosine_similarity(current_vec, prior_vecs)[0]


# -------------------- CORE --------------------

def extract_features(text):
    modalities = ["mri", "ct", "xray", "ultrasound", "pet"]
    body_parts = [
        "brain", "head", "chest", "abdomen", "pelvis",
        "spine", "knee", "shoulder", "hip"
    ]

    mod = [m for m in modalities if m in text]
    body = [b for b in body_parts if b in text]

    return mod, body


# -------------------- ROUTES --------------------

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: RequestBody):
    logging.info(f"Received {len(payload.cases)} cases")

    predictions = []

    for case in payload.cases:
        logging.info(f"Case {case.case_id} → {len(case.prior_studies)} priors")

        cur = normalize(case.current_study.study_description)
        cur_mod, cur_body = extract_features(cur)

        prior_texts = [normalize(p.study_description) for p in case.prior_studies]

        # ✅ ONE TF-IDF per case 
        sim_scores = compute_similarities(cur, prior_texts)

        for i, prior in enumerate(case.prior_studies):
            prev = prior_texts[i]
            prev_mod, prev_body = extract_features(prev)

            age = years_between(case.current_study.study_date, prior.study_date)

            same_body = set(cur_body).intersection(prev_body)
            same_modality = cur_mod == prev_mod

            relevant = False

            # Rule 1: strongest
            if same_modality and same_body:
                relevant = True

            # Rule 2: brain/head mapping
            elif ("brain" in cur and "head" in prev) or ("head" in cur and "brain" in prev):
                if age < 5:
                    relevant = True

            #  Rule 3: too old
            elif age > 12:
                relevant = False

            # Rule 4: semantic similarity
            elif sim_scores[i] > 0.55:
                relevant = True

            predictions.append({
                "case_id": case.case_id,
                "study_id": prior.study_id,
                "predicted_is_relevant": relevant
            })

    return {"predictions": predictions}