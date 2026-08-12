from pathlib import Path
from typing import Literal, TypedDict

import joblib
import pandas as pd
from langgraph.graph import END, StateGraph

from src.rag.generator import generate_answer


# --------------------------------------------------
# Load ML model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "careguard_random_forest.joblib"

risk_model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# LangGraph state
# --------------------------------------------------

class CareGuardState(TypedDict, total=False):
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


# --------------------------------------------------
# Safety guardrail
# --------------------------------------------------

def safety_check(state: CareGuardState):
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
                "CareGuard AI does not provide diagnosis, "
                "prescriptions, medication changes, or treatment advice. "
                "Please consult a qualified healthcare professional."
            ),
            "source": "CareGuard Safety Guardrail"
        }

    return {
        "blocked": False
    }


def safety_route(
    state: CareGuardState
) -> Literal["blocked", "continue"]:

    if state.get("blocked"):
        return "blocked"

    return "continue"


def blocked_node(state: CareGuardState):
    return {
        "answer": state["answer"],
        "source": state["source"]
    }


# --------------------------------------------------
# Router
# --------------------------------------------------

def route_request(state: CareGuardState):
    query = state["query"].lower()

    prediction_keywords = [
        "risk",
        "predict",
        "prediction",
        "patient risk",
        "utilization score"
    ]

    knowledge_keywords = [
        "why",
        "what",
        "explain",
        "factors",
        "healthcare utilization"
    ]

    if any(
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
    state: CareGuardState
) -> Literal["prediction", "rag"]:

    return state["route"]


# --------------------------------------------------
# Prediction node
# --------------------------------------------------

def prediction_node(state: CareGuardState):

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
            "source": "CareGuard ML Risk Model"
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
        "source": "CareGuard ML Risk Model",
        "risk_probability": risk_probability,
        "risk_level": risk_level,
        "prediction": prediction
    }


# --------------------------------------------------
# RAG node
# --------------------------------------------------

def rag_node(state: CareGuardState):

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

workflow = StateGraph(CareGuardState)

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
    "rag",
    rag_node
)


# --------------------------------------------------
# Graph entry point
# --------------------------------------------------

workflow.set_entry_point("safety")


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
        "rag": "rag"
    }
)

workflow.add_edge(
    "prediction",
    END
)

workflow.add_edge(
    "rag",
    END
)


# --------------------------------------------------
# Compile graph
# --------------------------------------------------

careguard_graph = workflow.compile()


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

    return careguard_graph.invoke(
        state
    )


# --------------------------------------------------
# Local tests
# --------------------------------------------------

if __name__ == "__main__":

    # ----------------------------------------------
    # Test 1: RAG route
    # ----------------------------------------------

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
    print(rag_question)

    print("\nROUTE:")
    print(rag_result.get("route"))

    print("\nANSWER:")
    print(rag_result.get("answer"))

    print("\nSOURCE:")
    print(rag_result.get("source"))


    # ----------------------------------------------
    # Test 2: Prediction route
    # ----------------------------------------------

    prediction_question = (
        "Predict this patient's healthcare utilization risk."
    )

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

    prediction_result = run_agent(
        prediction_question,
        patient_data=patient_data
    )

    print("\n" + "=" * 60)
    print("PREDICTION TEST")
    print("=" * 60)

    print("QUESTION:")
    print(prediction_question)

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


    # ----------------------------------------------
    # Test 3: Safety guardrail
    # ----------------------------------------------

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
    print(unsafe_question)

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