from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI


# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------

load_dotenv()

# OpenAI client reads OPENAI_API_KEY from environment variables.
client = OpenAI()

DB_PATH = os.getenv("DB_PATH", "data.db")


# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------

app = FastAPI(title="GenAI Food Fabrication Stage 1 (English-only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# DB helpers (SQLite)
# -----------------------------------------------------------------------------

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    conn = db_connect()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            prompt TEXT NOT NULL,
            raw_intent_json TEXT NOT NULL,
            normalized_requirement_json TEXT NOT NULL,
            defaults_applied_json TEXT NOT NULL,
            needs_clarification_json TEXT NOT NULL,
            confidence_json TEXT NOT NULL,
            version TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS shapes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            label TEXT NOT NULL,
            shape_json TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


@app.on_event("startup")
def _startup() -> None:
    db_init()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------------
# Types (enums)
# -----------------------------------------------------------------------------

ShapeMode = Literal["preset", "custom"]
ShapeName = Literal["circle", "square", "star", "rabbit", "dog", "cube", "sphere", "unknown"]

TargetUser = Literal[
    "Infant",
    "Toddler",
    "Kids",
    "Adult-Female",
    "Adult-Male",
    "Elder-Female",
    "Elder-Male",
    "unknown",
]

MealType = Literal["snack", "meal-light", "meal-regular", "meal-heavy", "unknown"]
Texture = Literal["soft", "normal", "crunchy", "unknown"]


# -----------------------------------------------------------------------------
# Request/Response models
# -----------------------------------------------------------------------------

class ParseRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)


class EvidenceItem(BaseModel):
    field: str
    quote: str


class RangeValue(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None


class RawIntent(BaseModel):
    # LLM should classify into enums wherever possible.
    # If unknown, set to null or "unknown" depending on the field type.

    # Shape
    shape_mode: Optional[ShapeMode] = None
    shape_preset_name: Optional[ShapeName] = None  # Only for preset mode
    shape_custom_text: Optional[str] = None        # Only for custom mode

    # User and meal type
    target_user: Optional[TargetUser] = None
    meal_type: Optional[MealType] = None

    # Nutrition ranges (explicit only, never invented)
    kcal: RangeValue = Field(default_factory=RangeValue)
    sugar_g: RangeValue = Field(default_factory=RangeValue)

    # Constraints and preferences
    # Examples: ["low_sugar", "no_sugar", "high_calorie", "no_nuts"]
    constraints_text: List[str] = Field(default_factory=list)

    # Optional structured constraints the LLM can classify
    texture: Optional[Texture] = None
    allergens: List[str] = Field(default_factory=list)

    # Evidence quotes copied from prompt for traceability
    evidence: List[EvidenceItem] = Field(default_factory=list)


class NormalizedShape(BaseModel):
    mode: ShapeMode = "preset"
    preset_name: ShapeName = "unknown"
    custom_text: Optional[str] = None
    custom_ref: Optional[str] = None  # future: file id, url, svg id, etc.


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
    version: str = "0.4"
    prompt: Dict[str, Any]
    raw_intent: RawIntent
    normalized_requirement: NormalizedRequirement
    defaults_applied: List[DefaultApplied] = Field(default_factory=list)
    needs_clarification: List[ClarificationItem] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=Confidence)


# -----------------------------------------------------------------------------
# LLM extraction
# -----------------------------------------------------------------------------

def extract_raw_intent_llm(prompt: str) -> RawIntent:
    """
    LLM produces schema aligned classification for all attributes.
    The server validates and applies deterministic post processing only.
    """

    system_instructions = (
        "You are a classifier for a food fabrication pipeline.\n"
        "The app is English only.\n\n"
        "Hard rules:\n"
        "1. Do NOT invent numbers. If kcal or sugar grams are not explicitly stated, leave them null.\n"
        "2. Do NOT apply defaults.\n"
        "3. If uncertain, use unknown or null.\n"
        "4. For every non null or non unknown decision, add at least one evidence quote copied from the prompt.\n\n"
        "Schema guidance:\n"
        "- shape_mode must be one of: preset, custom\n"
        "- shape_preset_name must be one of: circle, square, star, rabbit, dog, cube, sphere, unknown\n"
        "- If the shape is not one of the presets, use shape_mode=custom and put the free form description in shape_custom_text.\n"
        "- target_user must be one of: Infant, Toddler, Kids, Adult-Female, Adult-Male, Elder-Female, Elder-Male, unknown\n"
        "- meal_type must be one of: snack, meal-light, meal-regular, meal-heavy, unknown\n"
        "- texture must be one of: soft, normal, crunchy, unknown\n\n"
        "Constraints:\n"
        "- Use constraints_text tags when mentioned or implied, such as: low_sugar, no_sugar, high_calorie, no_nuts.\n"
        "- If nuts should be avoided, add allergens=['nuts'] and include evidence.\n"
        "- If you detect high calorie intent but no explicit kcal number, add constraints_text=['high_calorie'] (or include it).\n"
        "- kcal and sugar_g must only be filled if the prompt explicitly contains numbers.\n"
    )

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": prompt},
        ],
        text_format=RawIntent,
    )
    return response.output_parsed


def extract_raw_intent_llm_with_retry(prompt: str) -> RawIntent:
    """
    LLM extraction with a single retry for resilience.
    """
    try:
        return extract_raw_intent_llm(prompt)
    except Exception:
        stricter = (
            "Return ONLY valid JSON matching the RawIntent schema.\n"
            "No extra keys. If unknown, use unknown or null.\n"
            "Do NOT invent numbers.\n"
        )
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": stricter},
                {"role": "user", "content": prompt},
            ],
            text_format=RawIntent,
        )
        return response.output_parsed


def normalize_constraints(constraints: List[str]) -> List[str]:
    out: List[str] = []
    for c in constraints or []:
        c2 = (c or "").strip().lower()
        if not c2:
            continue

        aliases = {
            "low sugar": "low_sugar",
            "low_sugar": "low_sugar",
            "no sugar": "no_sugar",
            "sugar free": "no_sugar",
            "sugar-free": "no_sugar",
            "no_sugar": "no_sugar",
            "high calorie": "high_calorie",
            "high-calorie": "high_calorie",
            "high_calorie": "high_calorie",
            "no nuts": "no_nuts",
            "nut-free": "no_nuts",
            "nut free": "no_nuts",
            "no_nuts": "no_nuts",
        }

        tag = aliases.get(c2, c2)
        if tag not in out:
            out.append(tag)

    return out


# -----------------------------------------------------------------------------
# Deterministic normalization (raw_intent -> requirement)
# -----------------------------------------------------------------------------

def normalize_from_raw(prompt: str, raw: RawIntent) -> RequirementResponse:
    normalized = NormalizedRequirement()
    defaults_applied: List[DefaultApplied] = []
    needs: List[ClarificationItem] = []
    conf = Confidence()

    constraints = normalize_constraints(raw.constraints_text)

    # Shape
    if raw.shape_mode == "preset":
        normalized.shape.mode = "preset"
        normalized.shape.preset_name = raw.shape_preset_name or "unknown"

        if normalized.shape.preset_name == "unknown":
            conf.shape = 0.0
            needs.append(
                ClarificationItem(
                    field="shape.preset_name",
                    question="What preset shape do you want? For example: circle, square, star, rabbit.",
                )
            )
        else:
            conf.shape = 0.9

    elif raw.shape_mode == "custom":
        normalized.shape.mode = "custom"
        normalized.shape.preset_name = "unknown"
        normalized.shape.custom_text = (raw.shape_custom_text or "").strip() or None
        conf.shape = 0.6 if normalized.shape.custom_text else 0.0

        if not normalized.shape.custom_text:
            needs.append(
                ClarificationItem(
                    field="shape.custom_text",
                    question="Describe the custom shape you want in one sentence.",
                )
            )
        else:
            needs.append(
                ClarificationItem(
                    field="shape.custom_ref",
                    question="For custom shapes, will you provide an outline file later (such as SVG), or should we generate from text only?",
                )
            )
    else:
        conf.shape = 0.0
        needs.append(
            ClarificationItem(
                field="shape_mode",
                question="Do you want a preset shape (circle, square, star, rabbit) or a custom shape description?",
            )
        )

    # Target user
    if raw.target_user and raw.target_user != "unknown":
        normalized.target_user = raw.target_user
        conf.target_user = 0.85
    else:
        conf.target_user = 0.0
        needs.append(
            ClarificationItem(
                field="target_user",
                question="Who is this food for? For example: kids, adult male, elder female.",
            )
        )

    # Meal type
    if raw.meal_type and raw.meal_type != "unknown":
        normalized.meal_type = raw.meal_type
        conf.meal_type = 0.85
    else:
        conf.meal_type = 0.0

    # Allergens
    for a in raw.allergens or []:
        a2 = (a or "").strip().lower()
        if a2 and a2 not in normalized.constraints.allergens:
            normalized.constraints.allergens.append(a2)

    if "no_nuts" in constraints and "nuts" not in normalized.constraints.allergens:
        normalized.constraints.allergens.append("nuts")

    # Texture
    if raw.texture and raw.texture in ["soft", "normal", "crunchy", "unknown"]:
        normalized.constraints.texture = raw.texture
    else:
        normalized.constraints.texture = "unknown"

    # Nutrition ranges from LLM (explicit only)
    normalized.nutrition.kcal.min = raw.kcal.min
    normalized.nutrition.kcal.max = raw.kcal.max
    normalized.nutrition.sugar_g.min = raw.sugar_g.min
    normalized.nutrition.sugar_g.max = raw.sugar_g.max

    if (
        raw.kcal.min is not None
        or raw.kcal.max is not None
        or raw.sugar_g.min is not None
        or raw.sugar_g.max is not None
    ):
        conf.nutrition = 0.8
    else:
        conf.nutrition = 0.0

    # Clarifications driven by constraint tags
    if "no_sugar" in constraints and normalized.nutrition.sugar_g.max is None:
        needs.append(
            ClarificationItem(
                field="nutrition.sugar_g.max",
                question="You asked for no sugar. Should sugar_g.max be 0g, or do you mean very low sugar?",
            )
        )
        conf.nutrition = max(conf.nutrition, 0.3)

    if "low_sugar" in constraints and normalized.nutrition.sugar_g.max is None:
        needs.append(
            ClarificationItem(
                field="nutrition.sugar_g.max",
                question="You asked for low sugar. What is the maximum sugar in grams you want?",
            )
        )
        conf.nutrition = max(conf.nutrition, 0.3)

    if "high_calorie" in constraints and normalized.nutrition.kcal.max is None and normalized.nutrition.kcal.min is None:
        needs.append(
            ClarificationItem(
                field="nutrition.kcal.min",
                question="You asked for high calorie. What minimum kcal target do you want per serving?",
            )
        )
        conf.nutrition = max(conf.nutrition, 0.3)

    # Optional deterministic default
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
        conf.nutrition = max(conf.nutrition, 0.4)

    return RequirementResponse(
        prompt={"text": prompt},
        raw_intent=raw,
        normalized_requirement=normalized,
        defaults_applied=defaults_applied,
        needs_clarification=needs,
        confidence=conf,
    )


# -----------------------------------------------------------------------------
# Optional: fallback extractor (rule-based) for robustness
# -----------------------------------------------------------------------------

def extract_raw_intent_rules(prompt: str) -> RawIntent:
    tl = prompt.lower()
    raw = RawIntent()

    # Basic heuristic fallback only.
    if "kids" in tl or "children" in tl:
        raw.target_user = "Kids"
        raw.evidence.append(EvidenceItem(field="target_user", quote="kids/children"))

    if "grandmother" in tl or "grandma" in tl:
        raw.target_user = "Elder-Female"
        raw.evidence.append(EvidenceItem(field="target_user", quote="grandmother/grandma"))

    if "snack" in tl:
        raw.meal_type = "snack"
        raw.evidence.append(EvidenceItem(field="meal_type", quote="snack"))

    if "low sugar" in tl:
        raw.constraints_text.append("low_sugar")
    if "no sugar" in tl or "sugar-free" in tl:
        raw.constraints_text.append("no_sugar")
    if "high calorie" in tl:
        raw.constraints_text.append("high_calorie")
    if "no nuts" in tl or "nut-free" in tl:
        raw.constraints_text.append("no_nuts")
        raw.allergens.append("nuts")

    # Preset shapes
    if "rabbit" in tl or "bunny" in tl:
        raw.shape_mode = "preset"
        raw.shape_preset_name = "rabbit"
    elif "circle" in tl or "round" in tl:
        raw.shape_mode = "preset"
        raw.shape_preset_name = "circle"
    elif "star" in tl:
        raw.shape_mode = "preset"
        raw.shape_preset_name = "star"

    return raw


# -----------------------------------------------------------------------------
# Persistence endpoints
# -----------------------------------------------------------------------------

class CreateRunRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    save: bool = True


class CreateRunResponse(BaseModel):
    run_id: Optional[int] = None
    result: RequirementResponse


class ListRunsResponse(BaseModel):
    items: List[Dict[str, Any]]


class SaveShapeRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=120)
    shape: NormalizedShape


class SaveShapeResponse(BaseModel):
    shape_id: int


class ListShapesResponse(BaseModel):
    items: List[Dict[str, Any]]


def save_run_to_db(result: RequirementResponse) -> int:
    conn = db_connect()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO runs (
            created_at, prompt,
            raw_intent_json,
            normalized_requirement_json,
            defaults_applied_json,
            needs_clarification_json,
            confidence_json,
            version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now_iso(),
            result.prompt.get("text", ""),
            result.raw_intent.model_dump_json(),
            result.normalized_requirement.model_dump_json(),
            json.dumps([d.model_dump() for d in result.defaults_applied]),
            json.dumps([n.model_dump() for n in result.needs_clarification]),
            result.confidence.model_dump_json(),
            result.version,
        ),
    )
    conn.commit()
    run_id = int(cur.lastrowid)
    conn.close()
    return run_id


def save_shape_to_db(label: str, shape: NormalizedShape) -> int:
    conn = db_connect()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO shapes (created_at, label, shape_json)
        VALUES (?, ?, ?)
        """,
        (
            utc_now_iso(),
            label,
            shape.model_dump_json(),
        ),
    )
    conn.commit()
    shape_id = int(cur.lastrowid)
    conn.close()
    return shape_id


# -----------------------------------------------------------------------------
# Core endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "db_path": DB_PATH}


@app.post("/debug/extract", response_model=RawIntent)
def debug_extract(req: ParseRequest):
    return extract_raw_intent_llm_with_retry(req.prompt)


@app.post("/parse", response_model=RequirementResponse)
def parse(req: ParseRequest):
    try:
        raw = extract_raw_intent_llm_with_retry(req.prompt)
    except Exception:
        raw = extract_raw_intent_rules(req.prompt)
    return normalize_from_raw(req.prompt, raw)


@app.post("/runs/create", response_model=CreateRunResponse)
def create_run(req: CreateRunRequest):
    result = parse(ParseRequest(prompt=req.prompt))
    run_id: Optional[int] = None
    if req.save:
        run_id = save_run_to_db(result)
    return CreateRunResponse(run_id=run_id, result=result)


@app.get("/runs", response_model=ListRunsResponse)
def list_runs(limit: int = 50):
    limit = max(1, min(limit, 200))
    conn = db_connect()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, created_at, prompt, version,
               raw_intent_json,
               normalized_requirement_json,
               defaults_applied_json,
               needs_clarification_json,
               confidence_json
        FROM runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()

    items: List[Dict[str, Any]] = []
    for r in rows:
        items.append(
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "prompt": r["prompt"],
                "version": r["version"],
                "raw_intent": json.loads(r["raw_intent_json"]),
                "normalized_requirement": json.loads(r["normalized_requirement_json"]),
                "defaults_applied": json.loads(r["defaults_applied_json"]),
                "needs_clarification": json.loads(r["needs_clarification_json"]),
                "confidence": json.loads(r["confidence_json"]),
            }
        )

    return ListRunsResponse(items=items)


@app.post("/shapes", response_model=SaveShapeResponse)
def save_shape(req: SaveShapeRequest):
    shape_id = save_shape_to_db(req.label, req.shape)
    return SaveShapeResponse(shape_id=shape_id)


@app.get("/shapes", response_model=ListShapesResponse)
def list_shapes(limit: int = 50):
    limit = max(1, min(limit, 200))
    conn = db_connect()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, created_at, label, shape_json
        FROM shapes
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()

    items: List[Dict[str, Any]] = []
    for r in rows:
        items.append(
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "label": r["label"],
                "shape": json.loads(r["shape_json"]),
            }
        )

    return ListShapesResponse(items=items)
