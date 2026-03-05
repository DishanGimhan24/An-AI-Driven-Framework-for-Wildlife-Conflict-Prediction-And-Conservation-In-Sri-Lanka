from typing import List, Dict, Tuple
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier

# -------------------------------------------------
# Paths
# -------------------------------------------------
RISK_MODEL_PATH = "artifacts/model/risk_model_v4.cbm"
TYPE_MODEL_PATH = "artifacts/model/type_model_v3.joblib"
META_PATH = "artifacts/model/model_meta_v4.json"
DATASET_PATH = "artifacts/data/ml_dataset_expanded_forecastsafe.csv"

# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI(title="Wildlife Offence Prediction & Early Warning API (v4)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("LOADED: Backend/app/main.py (v4 CatBoost risk + v3 type + hotspots + risk level)")

# -------------------------------------------------
# Load meta (training schema)
# -------------------------------------------------
with open(META_PATH, "r") as f:
    meta = json.load(f)

CATEGORICAL_COLS = meta["categorical_features"]
NUMERIC_COLS = meta["numeric_features"]
ALL_FEATURES = CATEGORICAL_COLS + NUMERIC_COLS

# -------------------------------------------------
# Load models
# -------------------------------------------------
# Risk model: CatBoost .cbm
risk_model = CatBoostClassifier()
risk_model.load_model(RISK_MODEL_PATH)

# Type model: sklearn pipeline joblib
type_model = joblib.load(TYPE_MODEL_PATH)

# -------------------------------------------------
# Load dataset for history lookup
# -------------------------------------------------
df = pd.read_csv(DATASET_PATH)

df["year"] = pd.to_numeric(df.get("year"), errors="coerce").fillna(0).astype(int)
df["month_num"] = pd.to_numeric(df.get("month_num"), errors="coerce").fillna(0).astype(int)
df["total_cases"] = pd.to_numeric(df.get("total_cases"), errors="coerce").fillna(0.0).astype(float)
df["has_offence"] = pd.to_numeric(df.get("has_offence"), errors="coerce").fillna(0).astype(int).clip(0, 1)
df["top_offence"] = df.get("top_offence", "None").fillna("None").astype(str)
df["region"] = df.get("region", "Unknown").fillna("Unknown").astype(str)
df["location"] = df.get("location", "Unknown").fillna("Unknown").astype(str)

_lookup: Dict[Tuple[str, str, int, int], dict] = {}
for r in df.itertuples(index=False):
    _lookup[(r.region, r.location, int(r.year), int(r.month_num))] = {
        "total_cases": float(r.total_cases),
        "has_offence": int(r.has_offence),
        "top_offence": str(r.top_offence),
    }

ALL_LOCATIONS = sorted(
    df[["region", "location"]].drop_duplicates().itertuples(index=False, name=None),
    key=lambda x: (x[0], x[1])
)

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def sri_lanka_season(m: int) -> str:
    if m in (12, 1, 2):
        return "NE_monsoon"
    if m in (5, 6, 7, 8, 9):
        return "SW_monsoon"
    if m in (3, 4):
        return "Inter_MarApr"
    return "Inter_OctNov"

def month_cyc(m: int):
    return float(np.sin(2 * np.pi * m / 12)), float(np.cos(2 * np.pi * m / 12))

def prev_month(y: int, m: int, back: int):
    for _ in range(back):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return y, m

def get_row(region: str, location: str, y: int, m: int) -> dict:
    return _lookup.get((region, location, y, m), {
        "total_cases": 0.0,
        "has_offence": 0,
        "top_offence": "None",
    })

def compute_history(region: str, location: str, year: int, month: int) -> dict:
    past = {i: get_row(region, location, *prev_month(year, month, i)) for i in range(1, 8)}

    return {
        "lag_cases_1": past[1]["total_cases"],
        "lag_cases_2": past[2]["total_cases"],
        "lag_cases_3": past[3]["total_cases"],
        "lag_has_1": past[1]["has_offence"],
        "lag_has_2": past[2]["has_offence"],
        "lag_has_3": past[3]["has_offence"],
        "lag_top_1": past[1]["top_offence"],
        "lag_top_2": past[2]["top_offence"],
        "lag_top_3": past[3]["top_offence"],
        "roll_3_cases": float(np.mean([past[i]["total_cases"] for i in (1, 2, 3)])),
        "roll_6_cases": float(np.mean([past[i]["total_cases"] for i in range(1, 7)])),
        "trend_3": float(past[1]["total_cases"] - past[4]["total_cases"]),
        "trend_6": float(past[1]["total_cases"] - past[7]["total_cases"]),
    }

def risk_level_from_percent(risk_percent: float) -> str:
    if risk_percent < 30:
        return "Low"
    if risk_percent < 60:
        return "Medium"
    return "High"

def predict_for(region: str, location: str, year: int, month: int) -> dict:
    month_sin, month_cos = month_cyc(month)
    season = sri_lanka_season(month)
    hist = compute_history(region, location, year, month)

    row = {
        "region": region,
        "location": location,
        "season": season,
        "month_num": month,
        "month_sin": month_sin,
        "month_cos": month_cos,
        **hist,
    }

    X = pd.DataFrame([row])[ALL_FEATURES]

    risk_prob = float(risk_model.predict_proba(X)[0][1])
    risk_percent = round(risk_prob * 100, 2)
    risk_level = risk_level_from_percent(risk_percent)

    type_probs = type_model.predict_proba(X)[0]
    classes = type_model.named_steps["model"].classes_
    top3_idx = np.argsort(type_probs)[-3:][::-1]
    top3 = [str(classes[i]) for i in top3_idx]

    predicted_type = top3[0]
    if risk_level in ("Medium", "High") and predicted_type == "None":
        for t in top3:
            if t != "None":
                predicted_type = t
                break

    return {
        "location_id": f"{region}|{location}",
        "risk_percent": risk_percent,
        "risk_level": risk_level,
        "predicted_offence_type": predicted_type,
        "top3_offence_types": top3,
    }

# -------------------------------------------------
# Schemas
# -------------------------------------------------
class PredictRequest(BaseModel):
    region: str
    location: str
    year: int = Field(ge=2000, le=2100)
    month: int = Field(ge=1, le=12)

class PredictResponse(BaseModel):
    location_id: str
    risk_percent: float
    risk_level: str
    predicted_offence_type: str
    top3_offence_types: List[str]

class HotspotItem(BaseModel):
    rank: int
    region: str
    location: str
    risk_percent: float
    risk_level: str
    predicted_offence_type: str

class HotspotsResponse(BaseModel):
    year: int
    month: int
    top_k: int
    items: List[HotspotItem]

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/regions")
def regions():
    return {"regions": sorted(df["region"].dropna().unique().tolist())}

@app.get("/locations")
def locations(region: str):
    sub = df[df["region"] == region]
    return {"locations": sorted(sub["location"].dropna().unique().tolist())}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not ((df["region"] == req.region) & (df["location"] == req.location)).any():
        raise HTTPException(status_code=404, detail="Region/location not found. Use dropdowns from /regions and /locations.")

    out = predict_for(req.region, req.location, req.year, req.month)
    return PredictResponse(**out)

@app.get("/hotspots", response_model=HotspotsResponse)
def hotspots(
    year: int = Query(..., ge=2000, le=2100),
    month: int = Query(..., ge=1, le=12),
    top_k: int = Query(10, ge=1, le=100),
):
    results = []
    for region, location in ALL_LOCATIONS:
        out = predict_for(region, location, year, month)
        results.append({
            "region": region,
            "location": location,
            "risk_percent": out["risk_percent"],
            "risk_level": out["risk_level"],
            "predicted_offence_type": out["predicted_offence_type"],
        })

    results.sort(key=lambda r: r["risk_percent"], reverse=True)
    results = results[:top_k]

    items = []
    for i, r in enumerate(results, start=1):
        items.append(HotspotItem(
            rank=i,
            region=r["region"],
            location=r["location"],
            risk_percent=r["risk_percent"],
            risk_level=r["risk_level"],
            predicted_offence_type=r["predicted_offence_type"],
        ))

    return HotspotsResponse(year=year, month=month, top_k=top_k, items=items)
