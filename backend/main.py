from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import re

app = FastAPI(title="GenAI Food Fabrication API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ParseRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)

def detect_language(text: str) -> str:
    if re.search(r"[가-힣]", text):
        if re.search(r"[A-Za-z]", text):
            return "mixed"
        return "ko"
    return "en"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/parse")
def parse(req: ParseRequest):
    return {
        "version": "0.1",
        "prompt": {"text": req.prompt, "language": detect_language(req.prompt)},
        "raw_intent": {
            "shape_text": None,
            "nutrition_text": None,
            "target_user_text": None,
            "meal_type_text": None,
            "constraints_text": [],
            "evidence": []
        },
        "normalized_requirement": {
            "shape": {"family": "unknown", "name": "unknown"},
            "nutrition": {"kcal": {"min": None, "max": None}, "sugar_g": {"min": None, "max": None}},
            "target_user": "unknown",
            "meal_type": "unknown",
            "constraints": {"allergens": [], "texture": "unknown"}
        },
        "defaults_applied": [],
        "needs_clarification": [],
        "confidence": {"shape": 0.0, "target_user": 0.0, "meal_type": 0.0, "nutrition": 0.0}
    }
