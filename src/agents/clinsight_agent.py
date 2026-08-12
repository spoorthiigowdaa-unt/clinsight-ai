from pathlib import Path
from typing import Literal, TypedDict

import joblib
import pandas as pd
import shap
from langgraph.graph import END, StateGraph

from src.rag.generator import generate_answer


# --------------------------------------------------
# Load ML model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "careguard_random_forest.joblib"

risk_model = joblib.load(MODEL_PATH)

preprocessor = risk_model.named_steps["preprocessor"]
classifier = risk_model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(classifier)


# --------------------------------------------------
# LangGraph state
# --------------------------------------------------

class ClinSightState(TypedDict, total=False):
    query: str
    route: str
    answer: str
    source: str
    blocked: bool

    age: int
    gender: str
    race: str
    ethnicity: str

    hist_total_encounters: int
    hist_total_conditions: int
    hist_total_procedures: int
    hist_total_medications: int

    risk_probability: float
    risk_level: str
    prediction: int

    risk_drivers: list


# --------------------------------------------------
# Safety guardrail
# --------------------------------------------------

def safety_check(state: ClinSightState):
    query = state["query"].lower()

    unsafe_medical_keywords = [
        "diagnose me",
        "diagnosis",
        "what medication should i take",
        "what medicine should i take",
        "prescribe",
        "treatment should i take",
        "should i stop my medication",
        "dosage"
    ]

    if any(
        keyword in query
        for keyword in unsafe_medical_keywords
    ):
        return {
            "blocked": True,
            "answer": (
                "ClinSight AI does not provide diagnosis, "
                "prescriptions, medication changes, or treatment advice. "
                "Please consult a qualified healthcare professional."
            ),
            "source": "ClinSight Safety Guardrail"
        }

    return {
        "blocked": False
    }


def safety_route(
    state: ClinSightState
) -> Literal["blocked", "continue"]:

    if state.get("blocked"):
        return "blocked"

    return "continue"


def blocked_node(state: ClinSightState):
    return {
        "answer": state["answer"],
        "source": state["source"]
    }


# --------------------------------------------------
# Router
# --------------------------------------------------

def route_request(state: ClinSightState):
    query = state["query"].lower()

    prediction_keywords = [
        "predict",
        "prediction",
        "patient risk",
        "utilization score",
        "risk score"
    ]

    explanation_keywords = [
        "explain",
        "why is",
        "why was",
        "why did",
        "risk factors",
        "what influenced",
        "what caused",
        "prediction factors"
    ]

    knowledge_keywords = [
        "why",
        "what",
        "factors",
        "healthcare utilization",
        "medical care"
    ]

    # Check explanation first so questions like:
    # "Explain this patient's risk score"
    # are routed to the explainability engine.

    if any(
        keyword in query
        for keyword in explanation_keywords
    ):
        route = "explanation"

    elif any(
        keyword in query
        for keyword in prediction_keywords
    ):
        route = "prediction"

    elif any(
        keyword in query
        for keyword in knowledge_keywords
    ):
        route = "rag"

    else:
        route = "rag"

    return {
        "route": route
    }


def select_route(
    state: ClinSightState
) -> Literal["prediction", "rag", "explanation"]:

    return state["route"]


# --------------------------------------------------
# Prediction node
# --------------------------------------------------

def prediction_node(state: ClinSightState):

    required_fields = [
        "age",
        "gender",
        "race",
        "ethnicity",
        "hist_total_encounters",
        "hist_total_conditions",
        "hist_total_procedures",
        "hist_total_medications"
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in state
    ]

    if missing_fields:
        return {
            "answer": (
                "Patient feature data is required for risk prediction. "
                f"Missing fields: {', '.join(missing_fields)}"
            ),
            "source": "ClinSight ML Risk Model"
        }

    input_data = pd.DataFrame(
        [{
            "AGE": state["age"],
            "GENDER": state["gender"],
            "RACE": state["race"],
            "ETHNICITY": state["ethnicity"],
            "HIST_TOTAL_ENCOUNTERS": state["hist_total_encounters"],
            "HIST_TOTAL_CONDITIONS": state["hist_total_conditions"],
            "HIST_TOTAL_PROCEDURES": state["hist_total_procedures"],
            "HIST_TOTAL_MEDICATIONS": state["hist_total_medications"]
        }]
    )

    risk_probability = float(
        risk_model.predict_proba(input_data)[0, 1]
    )

    prediction = int(
        risk_probability >= 0.50
    )

    if risk_probability >= 0.70:
        risk_level = "HIGH"

    elif risk_probability >= 0.40:
        risk_level = "MODERATE"

    else:
        risk_level = "LOW"

    answer = (
        f"Predicted high-utilization risk is "
        f"{risk_probability * 100:.2f}% "
        f"({risk_level} risk)."
    )

    return {
        "answer": answer,
        "source": "ClinSight ML Risk Model",
        "risk_probability": risk_probability,
        "risk_level": risk_level,
        "prediction": prediction
    }


# --------------------------------------------------
# Explanation node
# --------------------------------------------------

def explanation_node(state: ClinSightState):

    required_fields = [
        "age",
        "gender",
        "race",
        "ethnicity",
        "hist_total_encounters",
        "hist_total_conditions",
        "hist_total_procedures",
        "hist_total_medications"
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in state
    ]

    if missing_fields:
        return {
            "answer": (
                "Patient information is required before "
                "the prediction can be explained."
            ),
            "source": "ClinSight Explainability Engine"
        }

    # --------------------------------------------------
    # Prepare patient input
    # --------------------------------------------------

    input_data = pd.DataFrame(
        [{
            "AGE": state["age"],
            "GENDER": state["gender"],
            "RACE": state["race"],
            "ETHNICITY": state["ethnicity"],
            "HIST_TOTAL_ENCOUNTERS": state["hist_total_encounters"],
            "HIST_TOTAL_CONDITIONS": state["hist_total_conditions"],
            "HIST_TOTAL_PROCEDURES": state["hist_total_procedures"],
            "HIST_TOTAL_MEDICATIONS": state["hist_total_medications"]
        }]
    )

    # --------------------------------------------------
    # Risk prediction
    # --------------------------------------------------

    risk_probability = float(
        risk_model.predict_proba(input_data)[0, 1]
    )

    prediction = int(
        risk_probability >= 0.50
    )

    if risk_probability >= 0.70:
        risk_level = "HIGH"

    elif risk_probability >= 0.40:
        risk_level = "MODERATE"

    else:
        risk_level = "LOW"

    # --------------------------------------------------
    # Transform input using trained preprocessor
    # --------------------------------------------------

    transformed_input = preprocessor.transform(
        input_data
    )

    transformed_df = pd.DataFrame(
        transformed_input,
        columns=feature_names
    )

    # --------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------

    shap_values = explainer(
        transformed_df
    )

    positive_class_values = (
        shap_values.values[0, :, 1]
    )

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": positive_class_values
    })

    explanation_df["impact"] = (
        explanation_df["shap_value"].abs()
    )

    top_features = (
        explanation_df
        .sort_values(
            "impact",
            ascending=False
        )
        .head(8)
    )

    # --------------------------------------------------
    # Human-readable labels
    # --------------------------------------------------

    feature_label_map = {
        "AGE": "Age",
        "HIST_TOTAL_ENCOUNTERS": "Historical Encounters",
        "HIST_TOTAL_CONDITIONS": "Historical Conditions",
        "HIST_TOTAL_PROCEDURES": "Historical Procedures",
        "HIST_TOTAL_MEDICATIONS": "Historical Medications",
    }

    seen_feature_groups = set()
    risk_drivers = []

    for _, row in top_features.iterrows():

        raw_feature = (
            row["feature"]
            .replace("num__", "")
            .replace("cat__", "")
        )

        # --------------------------------------------------
        # Gender
        # --------------------------------------------------

        if raw_feature.startswith("GENDER_"):
            feature_group = "GENDER"

            category = raw_feature.split(
                "_",
                1
            )[1]

            if category != state["gender"]:
                continue

            clean_feature = (
                "Gender: Female"
                if category == "F"
                else "Gender: Male"
            )

        # --------------------------------------------------
        # Race
        # --------------------------------------------------

        elif raw_feature.startswith("RACE_"):
            feature_group = "RACE"

            category = raw_feature.split(
                "_",
                1
            )[1]

            if (
                category.lower()
                != state["race"].lower()
            ):
                continue

            clean_feature = (
                f"Race: {category.title()}"
            )

        # --------------------------------------------------
        # Ethnicity
        # --------------------------------------------------

        elif raw_feature.startswith("ETHNICITY_"):
            feature_group = "ETHNICITY"

            category = raw_feature.split(
                "_",
                1
            )[1]

            if (
                category.lower()
                != state["ethnicity"].lower()
            ):
                continue

            readable_ethnicity = (
                "Non-Hispanic"
                if category.lower() == "nonhispanic"
                else category.title()
            )

            clean_feature = (
                f"Ethnicity: {readable_ethnicity}"
            )

        # --------------------------------------------------
        # Numerical features
        # --------------------------------------------------

        else:
            feature_group = raw_feature

            clean_feature = (
                feature_label_map.get(
                    raw_feature,
                    raw_feature
                    .replace("_", " ")
                    .title()
                )
            )

        # Avoid duplicate one-hot feature groups
        if feature_group in seen_feature_groups:
            continue

        seen_feature_groups.add(
            feature_group
        )

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
    # Build natural-language explanation
    # --------------------------------------------------

    driver_text = "; ".join(
        [
            f"{driver['feature']} "
            f"{driver['direction']}"
            for driver in risk_drivers
        ]
    )

    answer = (
        f"The patient's predicted healthcare utilization risk is "
        f"{risk_probability * 100:.2f}% "
        f"({risk_level} risk). "
        f"The most influential model factors are: "
        f"{driver_text}."
    )

    return {
        "answer": answer,
        "source": "ClinSight Explainability Engine",
        "risk_probability": risk_probability,
        "risk_level": risk_level,
        "prediction": prediction,
        "risk_drivers": risk_drivers
    }


# --------------------------------------------------
# RAG node
# --------------------------------------------------

def rag_node(state: ClinSightState):

    result = generate_answer(
        query=state["query"],
        top_k=3
    )

    return {
        "answer": result["answer"],
        "source": ", ".join(
            result["sources"]
        )
    }


# --------------------------------------------------
# Build LangGraph workflow
# --------------------------------------------------

workflow = StateGraph(
    ClinSightState
)

workflow.add_node(
    "safety",
    safety_check
)

workflow.add_node(
    "blocked",
    blocked_node
)

workflow.add_node(
    "router",
    route_request
)

workflow.add_node(
    "prediction",
    prediction_node
)

workflow.add_node(
    "explanation",
    explanation_node
)

workflow.add_node(
    "rag",
    rag_node
)


# --------------------------------------------------
# Graph entry point
# --------------------------------------------------

workflow.set_entry_point(
    "safety"
)


# --------------------------------------------------
# Safety routing
# --------------------------------------------------

workflow.add_conditional_edges(
    "safety",
    safety_route,
    {
        "blocked": "blocked",
        "continue": "router"
    }
)

workflow.add_edge(
    "blocked",
    END
)


# --------------------------------------------------
# Main routing
# --------------------------------------------------

workflow.add_conditional_edges(
    "router",
    select_route,
    {
        "prediction": "prediction",
        "rag": "rag",
        "explanation": "explanation"
    }
)

workflow.add_edge(
    "prediction",
    END
)

workflow.add_edge(
    "explanation",
    END
)

workflow.add_edge(
    "rag",
    END
)


# --------------------------------------------------
# Compile graph
# --------------------------------------------------

clinsight_graph = workflow.compile()


# --------------------------------------------------
# Public agent function
# --------------------------------------------------

def run_agent(
    query: str,
    patient_data: dict | None = None
):

    state = {
        "query": query
    }

    if patient_data:
        state.update(
            patient_data
        )

    return clinsight_graph.invoke(
        state
    )


# --------------------------------------------------
# Local tests
# --------------------------------------------------

if __name__ == "__main__":

    # --------------------------------------------------
    # Shared patient data
    # --------------------------------------------------

    patient_data = {
        "age": 58,
        "gender": "F",
        "race": "white",
        "ethnicity": "nonhispanic",
        "hist_total_encounters": 35,
        "hist_total_conditions": 12,
        "hist_total_procedures": 28,
        "hist_total_medications": 10
    }


    # --------------------------------------------------
    # Test 1: RAG route
    # --------------------------------------------------

    rag_question = (
        "Why might a patient have high healthcare utilization?"
    )

    rag_result = run_agent(
        rag_question
    )

    print("\n" + "=" * 60)
    print("RAG TEST")
    print("=" * 60)

    print("QUESTION:")
    print(
        rag_question
    )

    print("\nROUTE:")
    print(
        rag_result.get("route")
    )

    print("\nANSWER:")
    print(
        rag_result.get("answer")
    )

    print("\nSOURCE:")
    print(
        rag_result.get("source")
    )


    # --------------------------------------------------
    # Test 2: Prediction route
    # --------------------------------------------------

    prediction_question = (
        "Predict this patient's healthcare utilization risk."
    )

    prediction_result = run_agent(
        prediction_question,
        patient_data=patient_data
    )

    print("\n" + "=" * 60)
    print("PREDICTION TEST")
    print("=" * 60)

    print("QUESTION:")
    print(
        prediction_question
    )

    print("\nROUTE:")
    print(
        prediction_result.get("route")
    )

    print("\nANSWER:")
    print(
        prediction_result.get("answer")
    )

    print("\nSOURCE:")
    print(
        prediction_result.get("source")
    )

    print("\nRISK PROBABILITY:")
    print(
        prediction_result.get(
            "risk_probability"
        )
    )

    print("\nRISK LEVEL:")
    print(
        prediction_result.get(
            "risk_level"
        )
    )


    # --------------------------------------------------
    # Test 3: Explanation route with SHAP
    # --------------------------------------------------

    explanation_question = (
        "Explain why this patient has this risk score."
    )

    explanation_result = run_agent(
        explanation_question,
        patient_data=patient_data
    )

    print("\n" + "=" * 60)
    print("EXPLANATION TEST")
    print("=" * 60)

    print("QUESTION:")
    print(
        explanation_question
    )

    print("\nROUTE:")
    print(
        explanation_result.get("route")
    )

    print("\nANSWER:")
    print(
        explanation_result.get("answer")
    )

    print("\nSOURCE:")
    print(
        explanation_result.get("source")
    )

    print("\nRISK PROBABILITY:")
    print(
        explanation_result.get(
            "risk_probability"
        )
    )

    print("\nRISK LEVEL:")
    print(
        explanation_result.get(
            "risk_level"
        )
    )

    print("\nSHAP RISK DRIVERS:")

    for driver in explanation_result.get(
        "risk_drivers",
        []
    ):
        print(
            f"- {driver['feature']}: "
            f"{driver['direction']} "
            f"(importance={driver['importance']})"
        )


    # --------------------------------------------------
    # Test 4: Safety guardrail
    # --------------------------------------------------

    unsafe_question = (
        "What medication should I take for "
        "high healthcare utilization?"
    )

    unsafe_result = run_agent(
        unsafe_question
    )

    print("\n" + "=" * 60)
    print("SAFETY TEST")
    print("=" * 60)

    print("QUESTION:")
    print(
        unsafe_question
    )

    print("\nBLOCKED:")
    print(
        unsafe_result.get("blocked")
    )

    print("\nANSWER:")
    print(
        unsafe_result.get("answer")
    )

    print("\nSOURCE:")
    print(
        unsafe_result.get("source")
    )