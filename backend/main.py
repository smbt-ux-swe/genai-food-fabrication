from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

DB_PATH = os.getenv("DB_PATH", "app.db")


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
EatingOccasion = Literal[
    "breakfast",
    "lunch",
    "dinner",
    "snack_1",
    "snack_2",
    "snack",
    "unknown",
]
Texture = Literal["soft", "normal", "crunchy", "unknown"]

Sex = Literal["female", "male", "unknown"]

ActivityLevel = Literal[
    "confined",
    "ambulatory_light",
    "low",
    "active",
    "very_active",
    "unknown",
]

WeightGoal = Literal["maintain", "gain", "loss", "unknown"]


class EvidenceItem(BaseModel):
    field: str
    quote: str


class RangeValue(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None


class MacroPercent(BaseModel):
    carbs: float
    protein: float
    fat: float


class MacroGramsRange(BaseModel):
    carbs_g: RangeValue = Field(default_factory=RangeValue)
    protein_g: RangeValue = Field(default_factory=RangeValue)
    fat_g: RangeValue = Field(default_factory=RangeValue)


class NutritionComposition(BaseModel):
    method: str = "AMDR"
    macro_percent: MacroPercent
    macro_grams: MacroGramsRange


class RawIntent(BaseModel):
    shape_mode: Optional[ShapeMode] = None
    shape_preset_name: Optional[ShapeName] = None
    shape_custom_text: Optional[str] = None

    target_user: Optional[TargetUser] = None
    user_hint_text: Optional[str] = None

    sex: Optional[Sex] = None
    age_years: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    activity_level: Optional[ActivityLevel] = None
    illness_condition: Optional[str] = None
    weight_goal: Optional[WeightGoal] = None

    meal_type: Optional[MealType] = None
    eating_occasion: Optional[EatingOccasion] = None

    allergens: Optional[List[str]] = None
    texture: Optional[Texture] = None
    constraints_text: Optional[List[str]] = None

    kcal: Optional[float] = None
    sugar_g: Optional[float] = None

    evidence: List[EvidenceItem] = Field(default_factory=list)


class NormalizedShape(BaseModel):
    mode: ShapeMode = "preset"
    preset_name: ShapeName = "unknown"
    custom_text: Optional[str] = None


class NormalizedUserProfile(BaseModel):
    sex: Sex = "unknown"
    age_years: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    activity_level: ActivityLevel = "unknown"
    illness_condition: Optional[str] = None
    weight_goal: WeightGoal = "unknown"
    inferred_from: List[str] = Field(default_factory=list)


class NormalizedNutrition(BaseModel):
    kcal: RangeValue = Field(default_factory=RangeValue)
    sugar_g: RangeValue = Field(default_factory=RangeValue)


class NormalizedConstraints(BaseModel):
    allergens: List[str] = Field(default_factory=list)
    texture: Texture = "unknown"
    constraints_text: List[str] = Field(default_factory=list)


class NormalizedRequirement(BaseModel):
    shape: NormalizedShape = Field(default_factory=NormalizedShape)
    target_user: TargetUser = "unknown"
    user_profile: NormalizedUserProfile = Field(default_factory=NormalizedUserProfile)
    meal_type: MealType = "unknown"
    eating_occasion: EatingOccasion = "unknown"
    nutrition: NormalizedNutrition = Field(default_factory=NormalizedNutrition)
    constraints: NormalizedConstraints = Field(default_factory=NormalizedConstraints)


class DefaultApplied(BaseModel):
    field: str
    default: Dict[str, Any]
    reason: str


class ClarificationItem(BaseModel):
    field: str
    question: str


class ParseResult(BaseModel):
    raw_intent: RawIntent
    normalized_requirement: NormalizedRequirement
    defaults_applied: List[DefaultApplied] = Field(default_factory=list)
    clarifications: List[ClarificationItem] = Field(default_factory=list)


class ParseRequest(BaseModel):
    prompt: str


class CreateRunRequest(BaseModel):
    prompt: str
    save: bool = True


class CreateRunResponse(BaseModel):
    run_id: Optional[int]
    result: ParseResult


class ListRunsResponse(BaseModel):
    items: List[Dict[str, Any]]


class DietitianRunRequest(BaseModel):
    requirement: Dict[str, Any]
    use_kb: bool = True
    kb_path: str = "knowledgebases/dietitian_kb.md"


class DietitianCompareResponse(BaseModel):
    without_kb: Dict[str, Any]
    with_kb: Dict[str, Any]


def extract_raw(prompt: str) -> RawIntent:
    system_instructions = (
        "You extract structured intent for an AI Food Fabrication system.\n"
        "The app is English only.\n\n"
        "Hard rules:\n"
        "1. Do NOT invent numbers. If kcal or sugar grams are not explicitly stated, leave them null.\n"
        "2. Do NOT apply defaults.\n"
        "3. If uncertain, use unknown or null.\n"
        "4. For every non null or non unknown decision, add at least one evidence quote copied from the prompt.\n\n"
        "Schema guidance:\n"
        "- shape_mode must be one of: preset, custom\n"
        "- shape_preset_name must be one of: circle, square, star, rabbit, dog, cube, sphere, unknown\n"
        "- If the shape is not one of the presets, use shape_mode=custom and put free form text in shape_custom_text.\n"
        "- target_user must be one of: Infant, Toddler, Kids, Adult-Female, Adult-Male, Elder-Female, Elder-Male, unknown\n"
        "- sex must be one of: female, male, unknown\n"
        "- activity_level must be one of: confined, ambulatory_light, low, active, very_active, unknown\n"
        "- weight_goal must be one of: maintain, gain, loss, unknown\n"
        "- meal_type must be one of: snack, meal-light, meal-regular, meal-heavy, unknown\n"
        "- eating_occasion must be one of: breakfast, lunch, dinner, snack_1, snack_2, snack, unknown\n"
        "- texture must be one of: soft, normal, crunchy, unknown\n\n"
        "Notes:\n"
        "- user_hint_text can include words like grandma, grandpa, toddler, etc.\n"
        "- illness_condition should copy the user's text if present, otherwise null.\n"
        "- allergens should include foods the user says to avoid.\n"
        "- constraints_text can include high level intents like 'low_sugar' if explicitly requested.\n"
        "- If the user explicitly mentions breakfast, lunch, dinner, supper, or snack, set eating_occasion accordingly.\n"
        "- supper should map to dinner.\n"
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


def normalize_allergens(allergens: List[str]) -> List[str]:
    out: List[str] = []
    for a in allergens or []:
        a2 = (a or "").strip().lower()
        if not a2:
            continue
        out.append(a2)
    uniq: List[str] = []
    seen = set()
    for a in out:
        if a in seen:
            continue
        seen.add(a)
        uniq.append(a)
    return uniq


def normalize_constraints_text(constraints: List[str]) -> List[str]:
    out: List[str] = []
    for c in constraints or []:
        c2 = (c or "").strip().lower()
        if not c2:
            continue
        aliases = {
            "low sugar": "low_sugar",
            "less sugar": "low_sugar",
            "high protein": "high_protein",
            "low calorie": "low_calorie",
            "high calorie": "high_calorie",
        }
        out.append(aliases.get(c2, c2))
    uniq: List[str] = []
    seen = set()
    for c in out:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def infer_user_defaults(raw: RawIntent) -> Tuple[NormalizedUserProfile, List[DefaultApplied]]:
    defaults: List[DefaultApplied] = []
    profile = NormalizedUserProfile()

    if raw.sex:
        profile.sex = raw.sex
    if raw.age_years is not None:
        profile.age_years = raw.age_years
    if raw.height_cm is not None:
        profile.height_cm = raw.height_cm
    if raw.weight_kg is not None:
        profile.weight_kg = raw.weight_kg
    if raw.activity_level:
        profile.activity_level = raw.activity_level
    if raw.illness_condition:
        profile.illness_condition = raw.illness_condition
    if raw.weight_goal:
        profile.weight_goal = raw.weight_goal

    hint = (raw.user_hint_text or "").lower().strip()

    def apply_default(field: str, value: Dict[str, Any], reason: str):
        defaults.append(DefaultApplied(field=field, default=value, reason=reason))
        profile.inferred_from.append(field)

    if hint:
        if "grandma" in hint or "grandmother" in hint:
            if profile.sex == "unknown":
                profile.sex = "female"
                apply_default("user_profile.sex", {"sex": "female"}, "hint includes grandma")
            if profile.age_years is None:
                profile.age_years = 70
                apply_default("user_profile.age_years", {"age_years": 70}, "hint includes grandma")
        if "grandpa" in hint or "grandfather" in hint:
            if profile.sex == "unknown":
                profile.sex = "male"
                apply_default("user_profile.sex", {"sex": "male"}, "hint includes grandpa")
            if profile.age_years is None:
                profile.age_years = 70
                apply_default("user_profile.age_years", {"age_years": 70}, "hint includes grandpa")

    tu = raw.target_user or "unknown"
    if tu == "Elder-Female":
        if profile.sex == "unknown":
            profile.sex = "female"
            apply_default("user_profile.sex", {"sex": "female"}, "target_user is Elder-Female")
        if profile.age_years is None:
            profile.age_years = 70
            apply_default("user_profile.age_years", {"age_years": 70}, "target_user is Elder-Female")
    if tu == "Elder-Male":
        if profile.sex == "unknown":
            profile.sex = "male"
            apply_default("user_profile.sex", {"sex": "male"}, "target_user is Elder-Male")
        if profile.age_years is None:
            profile.age_years = 70
            apply_default("user_profile.age_years", {"age_years": 70}, "target_user is Elder-Male")
    if tu == "Adult-Female":
        if profile.sex == "unknown":
            profile.sex = "female"
            apply_default("user_profile.sex", {"sex": "female"}, "target_user is Adult-Female")
        if profile.age_years is None:
            profile.age_years = 35
            apply_default("user_profile.age_years", {"age_years": 35}, "target_user is Adult-Female")
    if tu == "Adult-Male":
        if profile.sex == "unknown":
            profile.sex = "male"
            apply_default("user_profile.sex", {"sex": "male"}, "target_user is Adult-Male")
        if profile.age_years is None:
            profile.age_years = 35
            apply_default("user_profile.age_years", {"age_years": 35}, "target_user is Adult-Male")

    if profile.activity_level == "unknown":
        profile.activity_level = "ambulatory_light"
        apply_default("user_profile.activity_level", {"activity_level": "ambulatory_light"}, "default activity level")

    if profile.weight_goal == "unknown":
        profile.weight_goal = "maintain"
        apply_default("user_profile.weight_goal", {"weight_goal": "maintain"}, "default weight goal")

    return profile, defaults


def normalize_from_raw(raw: RawIntent) -> ParseResult:
    defaults_applied: List[DefaultApplied] = []
    clarifications: List[ClarificationItem] = []

    shape = NormalizedShape()
    if raw.shape_mode:
        shape.mode = raw.shape_mode
    if shape.mode == "preset":
        shape.preset_name = raw.shape_preset_name or "unknown"
        shape.custom_text = None
    else:
        shape.preset_name = "unknown"
        shape.custom_text = raw.shape_custom_text

    profile, profile_defaults = infer_user_defaults(raw)
    defaults_applied.extend(profile_defaults)

    req = NormalizedRequirement(
        shape=shape,
        target_user=raw.target_user or "unknown",
        user_profile=profile,
        meal_type=raw.meal_type or "unknown",
        eating_occasion=raw.eating_occasion or "unknown",
    )

    req.constraints.allergens = normalize_allergens(raw.allergens or [])
    req.constraints.texture = raw.texture or "unknown"
    req.constraints.constraints_text = normalize_constraints_text(raw.constraints_text or [])

    if raw.kcal is not None:
        req.nutrition.kcal = RangeValue(min=float(raw.kcal), max=float(raw.kcal))
    if raw.sugar_g is not None:
        req.nutrition.sugar_g = RangeValue(min=float(raw.sugar_g), max=float(raw.sugar_g))

    if req.user_profile.weight_kg is None:
        clarifications.append(
            ClarificationItem(field="user_profile.weight_kg", question="What is the user's weight in kg?")
        )
    if req.user_profile.height_cm is None:
        clarifications.append(
            ClarificationItem(field="user_profile.height_cm", question="What is the user's height in cm?")
        )
    if req.user_profile.age_years is None:
        clarifications.append(
            ClarificationItem(field="user_profile.age_years", question="What is the user's age in years?")
        )
    if req.user_profile.sex == "unknown":
        clarifications.append(
            ClarificationItem(field="user_profile.sex", question="Is the user female or male?")
        )

    return ParseResult(
        raw_intent=raw,
        normalized_requirement=req,
        defaults_applied=defaults_applied,
        clarifications=clarifications,
    )


def parse(req: ParseRequest) -> ParseResult:
    raw = extract_raw(req.prompt)
    return normalize_from_raw(raw)


def load_kb_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def mifflin_st_jeor_rmr(weight_kg: float, height_cm: float, age_years: int, sex: Sex) -> float:
    if sex == "male":
        return (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age_years) + 5.0
    if sex == "female":
        return (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age_years) - 161.0
    return (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age_years) - 78.0


def activity_factor(level: ActivityLevel) -> float:
    mapping = {
        "confined": 1.1,
        "ambulatory_light": 1.3,
        "low": 1.5,
        "active": 1.75,
        "very_active": 1.9,
        "unknown": 1.3,
    }
    return mapping.get(level, 1.3)


def injury_stress_factor(illness: Optional[str]) -> float:
    if not illness:
        return 1.0
    t = illness.lower()
    if any(k in t for k in ["no illness", "none", "healthy"]):
        return 1.0
    if any(k in t for k in ["minor surgery", "mild", "moderate", "wound healing"]):
        return 1.2
    if any(k in t for k in ["sepsis"]):
        return 1.35
    if any(k in t for k in ["infection"]):
        return 1.3
    if any(k in t for k in ["fracture", "long-bone"]):
        return 1.3
    if any(k in t for k in ["closed head injury", "head injury"]):
        return 1.4
    if any(k in t for k in ["cancer"]):
        return 1.25
    if any(k in t for k in ["major surgery", "major trauma", "severe stress"]):
        return 1.6
    if any(k in t for k in ["burn"]):
        return 1.7
    if "fever" in t:
        return 1.2
    return 1.1


def meal_kcal_fraction(meal_type: MealType) -> float:
    mapping = {
        "snack": 0.15,
        "meal-light": 0.25,
        "meal-regular": 0.30,
        "meal-heavy": 0.35,
        "unknown": 0.25,
    }
    return mapping.get(meal_type, 0.25)


def cacfp_fraction_for_occasion(age_years: int, occasion: EatingOccasion) -> float:
    if age_years <= 4:
        mapping = {
            "breakfast": 0.20,
            "lunch": 0.26,
            "dinner": 0.26,
            "snack_1": 0.14,
            "snack_2": 0.14,
            "snack": 0.14,
            "unknown": 0.0,
        }
        return mapping.get(occasion, 0.0)

    mapping = {
        "breakfast": 0.22,
        "lunch": 0.32,
        "dinner": 0.32,
        "snack_1": 0.07,
        "snack_2": 0.07,
        "snack": 0.07,
        "unknown": 0.0,
    }
    return mapping.get(occasion, 0.0)


def meal_fraction(requirement: NormalizedRequirement) -> Tuple[float, str]:
    age = requirement.user_profile.age_years
    occ = requirement.eating_occasion

    if age is not None and occ != "unknown":
        f = cacfp_fraction_for_occasion(int(age), occ)
        if f > 0:
            return f, "cacfp_table_6_3"

    return meal_kcal_fraction(requirement.meal_type), "meal_type_fallback"


def default_macro_percent() -> MacroPercent:
    return MacroPercent(carbs=0.55, protein=0.20, fat=0.25)


def macro_grams_from_kcal(kcal: float, p: MacroPercent) -> Dict[str, float]:
    return {
        "carbs_g": (kcal * p.carbs) / 4.0,
        "protein_g": (kcal * p.protein) / 4.0,
        "fat_g": (kcal * p.fat) / 9.0,
    }


def macro_grams_range_from_kcal_range(kcal_min: float, kcal_max: float, p: MacroPercent) -> MacroGramsRange:
    gmin = macro_grams_from_kcal(kcal_min, p)
    gmax = macro_grams_from_kcal(kcal_max, p)

    return MacroGramsRange(
        carbs_g=RangeValue(min=round(gmin["carbs_g"], 1), max=round(gmax["carbs_g"], 1)),
        protein_g=RangeValue(min=round(gmin["protein_g"], 1), max=round(gmax["protein_g"], 1)),
        fat_g=RangeValue(min=round(gmin["fat_g"], 1), max=round(gmax["fat_g"], 1)),
    )


def recommend_texture(target_user: TargetUser, explicit_texture: Texture) -> Texture:
    if explicit_texture != "unknown":
        return explicit_texture
    if target_user in ["Infant", "Toddler"]:
        return "soft"
    if target_user in ["Elder-Female", "Elder-Male"]:
        return "soft"
    return "normal"


def recommend_size_grams(meal_type: MealType, target_user: TargetUser) -> RangeValue:
    if meal_type == "snack":
        if target_user in ["Infant", "Toddler"]:
            return RangeValue(min=15, max=35)
        if target_user in ["Kids"]:
            return RangeValue(min=25, max=50)
        return RangeValue(min=30, max=70)
    if meal_type == "meal-light":
        return RangeValue(min=80, max=180)
    if meal_type == "meal-regular":
        return RangeValue(min=150, max=300)
    if meal_type == "meal-heavy":
        return RangeValue(min=250, max=450)
    return RangeValue(min=80, max=180)


def dietitian_without_kb(requirement: NormalizedRequirement) -> Dict[str, Any]:
    meal = requirement.meal_type
    base_kcal = {
        "snack": 250,
        "meal-light": 450,
        "meal-regular": 650,
        "meal-heavy": 850,
        "unknown": 450,
    }.get(meal, 450)

    kcal_min = base_kcal * 0.9
    kcal_max = base_kcal * 1.1

    sugar_max = round(kcal_max * 0.10 / 4.0, 1)

    macro_percent = default_macro_percent()
    macro_grams = macro_grams_range_from_kcal_range(kcal_min, kcal_max, macro_percent)
    composition = NutritionComposition(
        method="AMDR",
        macro_percent=macro_percent,
        macro_grams=macro_grams,
    ).model_dump()

    out = {
        "nutrition_targets": {
            "kcal": {"min": round(kcal_min, 1), "max": round(kcal_max, 1)},
            "sugar_g": {"min": 0, "max": sugar_max},
            "composition": composition,
        },
        "constraints": {
            "texture": recommend_texture(requirement.target_user, requirement.constraints.texture),
            "size_g": recommend_size_grams(requirement.meal_type, requirement.target_user).model_dump(),
            "allergens": requirement.constraints.allergens,
        },
        "assumptions": [
            "Baseline heuristic without medical equations",
            "Sugar max set to 10% of kcal",
            "Macro split uses a default planning split within AMDR ranges",
        ],
        "missing_fields": [],
        "sources_used": [],
    }
    return out


def dietitian_with_kb(requirement: NormalizedRequirement, kb_path: str) -> Dict[str, Any]:
    profile = requirement.user_profile
    missing: List[str] = []
    if profile.weight_kg is None:
        missing.append("weight_kg")
    if profile.height_cm is None:
        missing.append("height_cm")
    if profile.age_years is None:
        missing.append("age_years")
    if profile.sex == "unknown":
        missing.append("sex")

    if missing:
        return {
            "nutrition_targets": {
                "kcal": requirement.nutrition.kcal.model_dump(),
                "sugar_g": requirement.nutrition.sugar_g.model_dump(),
            },
            "constraints": {
                "texture": recommend_texture(requirement.target_user, requirement.constraints.texture),
                "size_g": recommend_size_grams(requirement.meal_type, requirement.target_user).model_dump(),
                "allergens": requirement.constraints.allergens,
            },
            "assumptions": [
                "KB based method requires sex, age, height, weight",
                "Returned existing nutrition numbers if provided, otherwise left empty",
            ],
            "missing_fields": missing,
            "sources_used": ["dietitian_kb: required inputs section"],
        }

    weight_kg = float(profile.weight_kg)
    height_cm = float(profile.height_cm)
    age_years = int(profile.age_years)
    sex = profile.sex

    rmr = mifflin_st_jeor_rmr(weight_kg, height_cm, age_years, sex)
    af = activity_factor(profile.activity_level)
    sf = injury_stress_factor(profile.illness_condition)

    maintenance_kcal = rmr * af * sf

    adj = 0.0
    if profile.weight_goal == "gain":
        adj = 500.0
    if profile.weight_goal == "loss":
        adj = -350.0

    tdee = maintenance_kcal + adj

    frac, frac_source = meal_fraction(requirement)
    meal_kcal = tdee * frac

    kcal_min = meal_kcal * 0.9
    kcal_max = meal_kcal * 1.1
    sugar_max = (kcal_max * 0.10) / 4.0

    macro_percent = default_macro_percent()
    macro_grams = macro_grams_range_from_kcal_range(kcal_min, kcal_max, macro_percent)
    composition = NutritionComposition(
        method="AMDR",
        macro_percent=macro_percent,
        macro_grams=macro_grams,
    ).model_dump()

    calculation_trace = {
        "inputs": {
            "sex": sex,
            "age_years": age_years,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "activity_level": profile.activity_level,
            "illness_condition": profile.illness_condition,
            "weight_goal": profile.weight_goal,
            "meal_type": requirement.meal_type,
            "eating_occasion": requirement.eating_occasion,
        },
        "sources": {
            "kcal_allocation": frac_source,
            "macro_method": "amdr_default_split",
        },
        "equations": {
            "rmr": "RMR = (10*w) + (6.25*h) - (5*a) + (5 if male else -161)",
            "tdee": "TDEE = RMR * AF * SF + weight_goal_adjustment",
            "meal_kcal": "MealKcal = TDEE * fraction",
            "kcal_range": "KcalMin = MealKcal*0.9, KcalMax = MealKcal*1.1",
            "carbs_g": "CarbsG = (Kcal * carb_percent) / 4",
            "protein_g": "ProteinG = (Kcal * protein_percent) / 4",
            "fat_g": "FatG = (Kcal * fat_percent) / 9",
            "sugar_max_g": "SugarMaxG = (KcalMax * 0.10) / 4",
        },
        "steps": [
            {
                "step": "RMR",
                "value": round(rmr, 2),
                "substitution": (
                    f"RMR = 10*{weight_kg} + 6.25*{height_cm} - 5*{age_years} + "
                    f"({ '5' if sex == 'male' else '-161' }) = {round(rmr, 2)}"
                ),
            },
            {
                "step": "AF and SF",
                "value": {"activity_factor": af, "stress_factor": sf},
                "substitution": f"Maintenance = {round(rmr,2)} * {af} * {sf} = {round(maintenance_kcal,2)}",
            },
            {
                "step": "TDEE",
                "value": round(tdee, 2),
                "substitution": f"TDEE = {round(maintenance_kcal,2)} + ({adj}) = {round(tdee,2)}",
            },
            {
                "step": "Meal fraction",
                "value": frac,
                "substitution": f"Fraction source = {frac_source}, fraction = {frac}",
            },
            {
                "step": "Meal kcal",
                "value": round(meal_kcal, 2),
                "substitution": f"MealKcal = {round(tdee,2)} * {frac} = {round(meal_kcal,2)}",
            },
            {
                "step": "Kcal range",
                "value": {"min": round(kcal_min, 2), "max": round(kcal_max, 2)},
                "substitution": (
                    f"KcalMin = {round(meal_kcal,2)}*0.9 = {round(kcal_min,2)}, "
                    f"KcalMax = {round(meal_kcal,2)}*1.1 = {round(kcal_max,2)}"
                ),
            },
            {
                "step": "AMDR macros",
                "value": composition,
                "substitution": (
                    f"Carbs g range = [({round(kcal_min,2)}*{macro_percent.carbs})/4, ({round(kcal_max,2)}*{macro_percent.carbs})/4], "
                    f"Protein g range = [({round(kcal_min,2)}*{macro_percent.protein})/4, ({round(kcal_max,2)}*{macro_percent.protein})/4], "
                    f"Fat g range = [({round(kcal_min,2)}*{macro_percent.fat})/9, ({round(kcal_max,2)}*{macro_percent.fat})/9]"
                ),
            },
            {
                "step": "Sugar max",
                "value": round(sugar_max, 2),
                "substitution": f"SugarMaxG = ({round(kcal_max,2)}*0.10)/4 = {round(sugar_max,2)}",
            },
        ],
        "notes": [
            "If eating_occasion is present and age_years is present, CACFP allocation is used.",
            "Otherwise meal_type fraction fallback is used.",
            "Macro split uses a default planning split within AMDR ranges.",
        ],
    }

    kb_text = load_kb_text(kb_path)
    sources_used = []
    if kb_text:
        sources_used = [
            "dietitian_kb: Mifflin St Jeor equation",
            "dietitian_kb: activity factor table",
            "dietitian_kb: injury and stress factor table",
            "dietitian_kb: weight gain or loss adjustment guidance",
            "dietitian_kb: CACFP eating occasion calorie allocation",
            "dietitian_kb: AMDR macronutrient distribution",
        ]

    out = {
        "nutrition_targets": {
            "kcal": {"min": round(kcal_min, 1), "max": round(kcal_max, 1)},
            "sugar_g": {"min": 0, "max": round(sugar_max, 1)},
            "composition": composition,
            "debug": {
                "rmr": round(rmr, 1),
                "activity_factor": af,
                "stress_factor": sf,
                "estimated_tdee": round(tdee, 1),
                "meal_fraction": frac,
                "meal_fraction_source": frac_source,
            },
        },
        "constraints": {
            "texture": recommend_texture(requirement.target_user, requirement.constraints.texture),
            "size_g": recommend_size_grams(requirement.meal_type, requirement.target_user).model_dump(),
            "allergens": requirement.constraints.allergens,
        },
        "assumptions": [
            "RMR computed using Mifflin St Jeor equation",
            "Total energy estimated as RMR times activity factor times stress or injury factor",
            "Meal energy estimated using CACFP eating occasion allocation when available, otherwise meal_type fallback",
            "Macro split uses a default planning split within AMDR ranges",
            "Sugar cap estimated as 10% of kcal converted to grams",
            "Weight goal adjustment uses a simple midrange of recommended ranges",
        ],
        "missing_fields": [],
        "sources_used": sources_used,
        "calculation_trace": calculation_trace,
    }

    return out


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
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
            clarifications_json TEXT NOT NULL
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_run_to_db(result: ParseResult, prompt: str) -> int:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO runs (created_at, prompt, raw_intent_json, normalized_requirement_json, defaults_applied_json, clarifications_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            now_iso(),
            prompt,
            result.raw_intent.model_dump_json(),
            result.normalized_requirement.model_dump_json(),
            json.dumps([d.model_dump() for d in result.defaults_applied]),
            json.dumps([c.model_dump() for c in result.clarifications]),
        ),
    )
    conn.commit()
    run_id = int(cur.lastrowid)
    conn.close()
    return run_id


app = FastAPI(title="AI Food Fabrication Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_init()


@app.get("/health")
def health():
    return {"ok": True, "db_path": DB_PATH}


@app.post("/debug/extract", response_model=RawIntent)
def debug_extract(req: ParseRequest):
    return extract_raw(req.prompt)


@app.post("/parse", response_model=ParseResult)
def parse_endpoint(req: ParseRequest):
    return parse(req)


@app.post("/runs/create", response_model=CreateRunResponse)
def create_run(req: CreateRunRequest):
    result = parse(ParseRequest(prompt=req.prompt))
    run_id: Optional[int] = None
    if req.save:
        run_id = save_run_to_db(result, req.prompt)
    return CreateRunResponse(run_id=run_id, result=result)


@app.get("/runs", response_model=ListRunsResponse)
def list_runs(limit: int = 50):
    limit = max(1, min(limit, 200))
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, created_at, prompt FROM runs ORDER BY id DESC LIMIT ?",
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
            }
        )
    return ListRunsResponse(items=items)


@app.post("/agents/dietitian/run")
def run_dietitian(req: DietitianRunRequest):
    requirement = NormalizedRequirement.model_validate(req.requirement)
    if req.use_kb:
        return dietitian_with_kb(requirement, req.kb_path)
    return dietitian_without_kb(requirement)


@app.post("/agents/dietitian/compare", response_model=DietitianCompareResponse)
def compare_dietitian(req: DietitianRunRequest):
    requirement = NormalizedRequirement.model_validate(req.requirement)
    without_kb = dietitian_without_kb(requirement)
    with_kb = dietitian_with_kb(requirement, req.kb_path)
    return DietitianCompareResponse(without_kb=without_kb, with_kb=with_kb)


class CreateShapeRequest(BaseModel):
    label: str
    shape: Dict[str, Any]


class CreateShapeResponse(BaseModel):
    id: int


class ListShapesResponse(BaseModel):
    items: List[Dict[str, Any]]


def save_shape_to_db(label: str, shape_obj: Dict[str, Any]) -> int:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO shapes (created_at, label, shape_json)
        VALUES (?, ?, ?)
        """,
        (
            now_iso(),
            label,
            json.dumps(shape_obj),
        ),
    )
    conn.commit()
    shape_id = int(cur.lastrowid)
    conn.close()
    return shape_id


@app.post("/shapes", response_model=CreateShapeResponse)
def create_shape(req: CreateShapeRequest):
    shape_id = save_shape_to_db(req.label, req.shape)
    return CreateShapeResponse(id=shape_id)


@app.get("/shapes", response_model=ListShapesResponse)
def list_shapes(limit: int = 50):
    limit = max(1, min(limit, 200))
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, created_at, label, shape_json FROM shapes ORDER BY id DESC LIMIT ?",
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
