from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import re

app = FastAPI(title="GenAI Food Fabrication Stage 1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ShapeFamily = Literal["simple", "advanced", "three_d", "unknown"]
ShapeName = Literal["circle", "square", "star", "rabbit", "dog", "cube", "sphere", "unknown"]
TargetUser = Literal[
    "Infant", "Toddler", "Kids",
    "Adult-Female", "Adult-Male",
    "Elder-Female", "Elder-Male",
    "unknown",
]
MealType = Literal["snack", "meal-light", "meal-regular", "meal-heavy", "unknown"]
Texture = Literal["soft", "normal", "crunchy", "unknown"]

class ParseRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)

class EvidenceItem(BaseModel):
    field: str
    quote: str

class RawIntent(BaseModel):
    shape_text: Optional[str] = None
    nutrition_text: Optional[str] = None
    target_user_text: Optional[str] = None
    meal_type_text: Optional[str] = None
    constraints_text: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)

class RangeValue(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None

class NormalizedShape(BaseModel):
    family: ShapeFamily = "unknown"
    name: ShapeName = "unknown"

class NormalizedNutrition(BaseModel):
    kcal: RangeValue = Field(default_factory=RangeValue)
    sugar_g: RangeValue = Field(default_factory=RangeValue)

class NormalizedConstraints(BaseModel):
    allergens: List[str] = Field(default_factory=list)
    texture: Texture = "unknown"

class NormalizedRequirement(BaseModel):
    shape: NormalizedShape = Field(default_factory=NormalizedShape)
    nutrition: NormalizedNutrition = Field(default_factory=NormalizedNutrition)
    target_user: TargetUser = "unknown"
    meal_type: MealType = "unknown"
    constraints: NormalizedConstraints = Field(default_factory=NormalizedConstraints)

class DefaultApplied(BaseModel):
    field: str
    default: Dict[str, Any]
    reason: str

class ClarificationItem(BaseModel):
    field: str
    question: str

class Confidence(BaseModel):
    shape: float = 0.0
    target_user: float = 0.0
    meal_type: float = 0.0
    nutrition: float = 0.0

class RequirementResponse(BaseModel):
    version: str = "0.1"
    prompt: Dict[str, Any]
    raw_intent: RawIntent
    normalized_requirement: NormalizedRequirement
    defaults_applied: List[DefaultApplied] = Field(default_factory=list)
    needs_clarification: List[ClarificationItem] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=Confidence)

def detect_language(text: str) -> str:
    if re.search(r"[가-힣]", text):
        if re.search(r"[A-Za-z]", text):
            return "mixed"
        return "ko"
    return "en"

def extract_kcal_range(text: str) -> Optional[tuple[float, float, str]]:
    t = text.lower()

    m = re.search(r"(\d+)\s*[-~]\s*(\d+)\s*kcal", t)
    if m:
        return float(m.group(1)), float(m.group(2)), m.group(0)

    m = re.search(r"(\d+)\s*kcal", t)
    if m:
        v = float(m.group(1))
        return v, v, m.group(0)

    m = re.search(r"(\d+)\s*[-~]\s*(\d+)\s*cal", t)
    if m:
        return float(m.group(1)), float(m.group(2)), m.group(0)

    m = re.search(r"(\d+)\s*cal", t)
    if m:
        v = float(m.group(1))
        return v, v, m.group(0)

    return None

def extract_sugar_range_grams(text: str) -> Optional[tuple[float, float, str]]:
    t = text.lower()

    m = re.search(r"(\d+)\s*[-~]\s*(\d+)\s*g\s*(?:sugar|sugars)", t)
    if m:
        return float(m.group(1)), float(m.group(2)), m.group(0)

    m = re.search(r"(\d+)\s*g\s*(?:sugar|sugars)", t)
    if m:
        v = float(m.group(1))
        return v, v, m.group(0)

    m = re.search(r"당\s*(\d+)\s*[-~]\s*(\d+)\s*g", text)
    if m:
        return float(m.group(1)), float(m.group(2)), m.group(0)

    m = re.search(r"당\s*(\d+)\s*g", text)
    if m:
        v = float(m.group(1))
        return v, v, m.group(0)

    return None

def normalize(prompt: str) -> RequirementResponse:
    lang = detect_language(prompt)
    tl = prompt.lower()

    raw = RawIntent()
    normalized = NormalizedRequirement()
    defaults_applied: List[DefaultApplied] = []
    needs: List[ClarificationItem] = []
    conf = Confidence()

    shape_map: Dict[str, List[str]] = {
        "rabbit": ["rabbit", "bunny", "토끼"],
        "dog": ["dog", "puppy", "강아지", "개"],
        "circle": ["circle", "원", "동그라미"],
        "square": ["square", "네모", "사각", "사각형"],
        "star": ["star", "별"],
        "cube": ["cube", "큐브", "정육면체"],
        "sphere": ["sphere", "구", "구체"],
    }

    found_shape: Optional[str] = None
    found_shape_quote: Optional[str] = None
    for shape_name, keys in shape_map.items():
        for k in keys:
            if k.lower() in tl:
                found_shape = shape_name
                found_shape_quote = k
                break
        if found_shape:
            break

    if found_shape:
        raw.shape_text = found_shape
        raw.evidence.append(EvidenceItem(field="shape_text", quote=found_shape_quote or found_shape))

        normalized.shape.name = found_shape  # type: ignore
        if found_shape in ["circle", "square", "star"]:
            normalized.shape.family = "simple"
        elif found_shape in ["rabbit", "dog"]:
            normalized.shape.family = "advanced"
        else:
            normalized.shape.family = "three_d"

        conf.shape = 0.9
    else:
        conf.shape = 0.0

    target_map: Dict[str, List[str]] = {
        "Infant": ["infant", "newborn", "baby", "신생아", "영아"],
        "Toddler": ["toddler", "유아", "아기"],
        "Kids": ["kids", "kid", "child", "children", "어린이", "아이"],
        "Adult-Male": ["adult male", "man", "male adult", "남자", "남성"],
        "Adult-Female": ["adult female", "woman", "female adult", "여자", "여성"],
        "Elder-Male": ["elder male", "senior man", "노인 남성", "할아버지"],
        "Elder-Female": ["elder female", "senior woman", "노인 여성", "할머니"],
    }

    found_user: Optional[str] = None
    found_user_quote: Optional[str] = None
    for user, keys in target_map.items():
        for k in keys:
            if k.lower() in tl:
                found_user = user
                found_user_quote = k
                break
        if found_user:
            break

    if found_user:
        raw.target_user_text = found_user
        raw.evidence.append(EvidenceItem(field="target_user_text", quote=found_user_quote or found_user))

        normalized.target_user = found_user  # type: ignore
        conf.target_user = 0.8
    else:
        conf.target_user = 0.0

    meal_map: Dict[str, List[str]] = {
        "snack": ["snack", "간식"],
        "meal-light": ["light meal", "가벼운 식사", "가벼운"],
        "meal-regular": ["meal", "lunch", "dinner", "식사", "점심", "저녁"],
        "meal-heavy": ["heavy meal", "high calorie", "든든", "배부르게"],
    }

    found_meal: Optional[str] = None
    found_meal_quote: Optional[str] = None
    for meal, keys in meal_map.items():
        for k in keys:
            if k.lower() in tl:
                found_meal = meal
                found_meal_quote = k
                break
        if found_meal:
            break

    if found_meal:
        raw.meal_type_text = found_meal
        raw.evidence.append(EvidenceItem(field="meal_type_text", quote=found_meal_quote or found_meal))

        normalized.meal_type = found_meal  # type: ignore
        conf.meal_type = 0.8
    else:
        conf.meal_type = 0.0

    constraints: List[str] = []

    if "low sugar" in tl or "저당" in tl or "당 줄" in tl:
        constraints.append("low_sugar")
    if "no sugar" in tl or "sugar-free" in tl or "무당" in tl:
        constraints.append("no_sugar")

    if "no nuts" in tl or "nut-free" in tl or "견과" in tl:
        constraints.append("no_nuts")

    if "soft" in tl or "부드" in tl:
        constraints.append("soft_texture")
    if "crunchy" in tl or "바삭" in tl:
        constraints.append("crunchy_texture")

    raw.constraints_text = constraints

    if "no_nuts" in constraints:
        normalized.constraints.allergens.append("nuts")

    if "soft_texture" in constraints and "crunchy_texture" in constraints:
        normalized.constraints.texture = "unknown"
        needs.append(
            ClarificationItem(
                field="constraints.texture",
                question="You mentioned both soft and crunchy. Which texture should it be?",
            )
        )
    elif "soft_texture" in constraints:
        normalized.constraints.texture = "soft"
    elif "crunchy_texture" in constraints:
        normalized.constraints.texture = "crunchy"

    kcal = extract_kcal_range(prompt)
    if kcal:
        kmin, kmax, quote = kcal
        raw.nutrition_text = (raw.nutrition_text + " | " if raw.nutrition_text else "") + quote
        raw.evidence.append(EvidenceItem(field="nutrition_text", quote=quote))

        normalized.nutrition.kcal.min = kmin
        normalized.nutrition.kcal.max = kmax
        conf.nutrition = 0.7

    sugar = extract_sugar_range_grams(prompt)
    if sugar:
        smin, smax, quote = sugar
        raw.nutrition_text = (raw.nutrition_text + " | " if raw.nutrition_text else "") + quote
        raw.evidence.append(EvidenceItem(field="nutrition_text", quote=quote))

        normalized.nutrition.sugar_g.min = smin
        normalized.nutrition.sugar_g.max = smax
        conf.nutrition = max(conf.nutrition, 0.7)

    if conf.nutrition == 0.0 and ("low_sugar" in constraints or "no_sugar" in constraints):
        conf.nutrition = 0.3

    if "no_sugar" in constraints and normalized.nutrition.sugar_g.max is None:
        needs.append(
            ClarificationItem(
                field="nutrition.sugar_g.max",
                question="You asked for no sugar. Should sugar_g.max be 0g, or do you mean very low sugar?",
            )
        )

    if "low_sugar" in constraints and normalized.nutrition.sugar_g.max is None:
        needs.append(
            ClarificationItem(
                field="nutrition.sugar_g.max",
                question="You asked for low sugar. What is the maximum sugar in grams you want?",
            )
        )

    if normalized.meal_type == "snack" and normalized.nutrition.kcal.min is None and normalized.nutrition.kcal.max is None:
        defaults_applied.append(
            DefaultApplied(
                field="nutrition.kcal",
                default={"min": 250, "max": 350},
                reason="Default kcal range for snack when not specified in prompt",
            )
        )
        normalized.nutrition.kcal.min = 250
        normalized.nutrition.kcal.max = 350

    if normalized.target_user == "unknown":
        needs.append(
            ClarificationItem(
                field="target_user",
                question="Who is this food for? Infant, toddler, kids, or adult?",
            )
        )

    if normalized.shape.name == "unknown":
        needs.append(
            ClarificationItem(
                field="shape.name",
                question="What shape do you want? For example circle, square, star, rabbit.",
            )
        )

    return RequirementResponse(
        prompt={"text": prompt, "language": lang},
        raw_intent=raw,
        normalized_requirement=normalized,
        defaults_applied=defaults_applied,
        needs_clarification=needs,
        confidence=conf,
    )

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/parse", response_model=RequirementResponse)
def parse(req: ParseRequest):
    return normalize(req.prompt)
