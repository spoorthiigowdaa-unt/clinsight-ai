from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agents.clinsight_agent import run_agent
from src.rag.generator import generate_answer


app = FastAPI(
    title="Clinsight AI API",
    description=(
        "Healthcare utilization risk prediction, explainability, "
        "RAG, and LangGraph agent API"
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Load model and explainability components
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "careguard_random_forest.joblib"

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(classifier)


# --------------------------------------------------
# Request schemas
# --------------------------------------------------

class PatientFeatures(BaseModel):
    age: int = Field(..., ge=0, le=120)
    gender: str
    race: str
    ethnicity: str
    hist_total_encounters: int = Field(..., ge=0)
    hist_total_conditions: int = Field(..., ge=0)
    hist_total_procedures: int = Field(..., ge=0)
    hist_total_medications: int = Field(..., ge=0)


class RagQuery(BaseModel):
    question: str = Field(..., min_length=3)


class AgentQuery(BaseModel):
    query: str = Field(..., min_length=3)

    age: int | None = None
    gender: str | None = None
    race: str | None = None
    ethnicity: str | None = None

    hist_total_encounters: int | None = None
    hist_total_conditions: int | None = None
    hist_total_procedures: int | None = None
    hist_total_medications: int | None = None


# --------------------------------------------------
# Basic API endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "application": "Clinsight AI",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(patient: PatientFeatures):

    # --------------------------------------------------
    # Prepare model input
    # --------------------------------------------------

    input_data = pd.DataFrame(
        [{
            "AGE": patient.age,
            "GENDER": patient.gender,
            "RACE": patient.race,
            "ETHNICITY": patient.ethnicity,
            "HIST_TOTAL_ENCOUNTERS": patient.hist_total_encounters,
            "HIST_TOTAL_CONDITIONS": patient.hist_total_conditions,
            "HIST_TOTAL_PROCEDURES": patient.hist_total_procedures,
            "HIST_TOTAL_MEDICATIONS": patient.hist_total_medications
        }]
    )

    # --------------------------------------------------
    # Risk prediction
    # --------------------------------------------------

    risk_probability = float(
        model.predict_proba(input_data)[0, 1]
    )

    prediction = int(risk_probability >= 0.50)

    if risk_probability >= 0.70:
        risk_level = "HIGH"
    elif risk_probability >= 0.40:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    # --------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------

    transformed_input = preprocessor.transform(input_data)

    transformed_df = pd.DataFrame(
        transformed_input,
        columns=feature_names
    )

    shap_values = explainer(transformed_df)

    positive_class_values = shap_values.values[0, :, 1]

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": positive_class_values
    })

    explanation_df["impact"] = explanation_df["shap_value"].abs()

    top_features = (
        explanation_df
        .sort_values("impact", ascending=False)
        .head(8)
    )

    risk_drivers = []

    # --------------------------------------------------
    # Human-readable feature labels
    # --------------------------------------------------

    feature_label_map = {
        "AGE": "Age",
        "HIST_TOTAL_ENCOUNTERS": "Historical Encounters",
        "HIST_TOTAL_CONDITIONS": "Historical Conditions",
        "HIST_TOTAL_PROCEDURES": "Historical Procedures",
        "HIST_TOTAL_MEDICATIONS": "Historical Medications",
    }

    seen_feature_groups = set()

    for _, row in top_features.iterrows():

        raw_feature = (
            row["feature"]
            .replace("num__", "")
            .replace("cat__", "")
        )

        if raw_feature.startswith("GENDER_"):
            feature_group = "GENDER"

            category = raw_feature.split("_", 1)[1]

            if category != patient.gender:
                continue

            clean_feature = (
                "Gender: Female"
                if category == "F"
                else "Gender: Male"
            )

        elif raw_feature.startswith("RACE_"):
            feature_group = "RACE"

            category = raw_feature.split("_", 1)[1]

            if category.lower() != patient.race.lower():
                continue

            clean_feature = f"Race: {category.title()}"

        elif raw_feature.startswith("ETHNICITY_"):
            feature_group = "ETHNICITY"

            category = raw_feature.split("_", 1)[1]

            if category.lower() != patient.ethnicity.lower():
                continue

            readable_ethnicity = (
                "Non-Hispanic"
                if category.lower() == "nonhispanic"
                else category.title()
            )

            clean_feature = f"Ethnicity: {readable_ethnicity}"

        else:
            feature_group = raw_feature

            clean_feature = feature_label_map.get(
                raw_feature,
                raw_feature.replace("_", " ").title()
            )

        if feature_group in seen_feature_groups:
            continue

        seen_feature_groups.add(feature_group)

        direction = (
            "increases risk"
            if row["shap_value"] > 0
            else "decreases risk"
        )

        risk_drivers.append({
            "feature": clean_feature,
            "direction": direction,
            "importance": round(
                float(row["impact"]),
                4
            )
        })

        if len(risk_drivers) == 5:
            break

    # --------------------------------------------------
    # Prediction API response
    # --------------------------------------------------

    return {
        "high_utilization_prediction": prediction,
        "risk_probability": round(
            risk_probability,
            4
        ),
        "risk_percentage": round(
            risk_probability * 100,
            2
        ),
        "risk_level": risk_level,
        "risk_drivers": risk_drivers
    }


# --------------------------------------------------
# RAG endpoint
# --------------------------------------------------

@app.post("/rag/query")
def rag_query(payload: RagQuery):

    result = generate_answer(
        query=payload.question,
        top_k=3
    )

    return {
        "question": payload.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }


# --------------------------------------------------
# LangGraph Agent endpoint
# --------------------------------------------------

@app.post("/agent/query")
def agent_query(payload: AgentQuery):

    patient_data = {
        "age": payload.age,
        "gender": payload.gender,
        "race": payload.race,
        "ethnicity": payload.ethnicity,
        "hist_total_encounters": payload.hist_total_encounters,
        "hist_total_conditions": payload.hist_total_conditions,
        "hist_total_procedures": payload.hist_total_procedures,
        "hist_total_medications": payload.hist_total_medications
    }

    patient_data = {
        key: value
        for key, value in patient_data.items()
        if value is not None
    }

    result = run_agent(
        query=payload.query,
        patient_data=patient_data
    )

    return {
        "query": payload.query,
        "route": result.get("route"),
        "answer": result.get("answer"),
        "source": result.get("source"),
        "risk_probability": result.get("risk_probability"),
        "risk_level": result.get("risk_level"),
        "prediction": result.get("prediction")
    }