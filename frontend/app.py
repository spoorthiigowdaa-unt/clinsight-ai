import os

import requests
import streamlit as st


API_BASE_URL = os.getenv(
    "CLINSIGHT_API_URL",
    "http://127.0.0.1:8000"
)

API_URL = f"{API_BASE_URL}/predict"
RAG_URL = f"{API_BASE_URL}/rag/query"
AGENT_URL = f"{API_BASE_URL}/agent/query"


st.set_page_config(
    page_title="ClinSight AI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 ClinSight AI")

st.subheader("Healthcare Utilization Risk Intelligence")

st.write(
    "Estimate a patient's future healthcare utilization risk "
    "using historical clinical and utilization information."
)

st.divider()


# --------------------------------------------------
# Patient demographics
# --------------------------------------------------

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=50
    )

    gender = st.selectbox(
        "Gender",
        ["F", "M"]
    )


with col2:
    race = st.selectbox(
        "Race",
        [
            "white",
            "black",
            "asian",
            "native",
            "other"
        ]
    )

    ethnicity = st.selectbox(
        "Ethnicity",
        [
            "nonhispanic",
            "hispanic"
        ]
    )


# --------------------------------------------------
# Historical utilization
# --------------------------------------------------

st.header("Historical Healthcare Activity")

col3, col4 = st.columns(2)

with col3:
    encounters = st.number_input(
        "Historical Encounters",
        min_value=0,
        value=10
    )

    conditions = st.number_input(
        "Historical Conditions",
        min_value=0,
        value=5
    )


with col4:
    procedures = st.number_input(
        "Historical Procedures",
        min_value=0,
        value=5
    )

    medications = st.number_input(
        "Historical Medications",
        min_value=0,
        value=3
    )


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Analyze Patient Risk",
    type="primary",
    use_container_width=True
):

    patient_data = {
        "age": age,
        "gender": gender,
        "race": race,
        "ethnicity": ethnicity,
        "hist_total_encounters": encounters,
        "hist_total_conditions": conditions,
        "hist_total_procedures": procedures,
        "hist_total_medications": medications
    }

    try:

        response = requests.post(
            API_URL,
            json=patient_data,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        st.header("Risk Assessment")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Risk Probability",
                f"{result['risk_percentage']}%"
            )

        with metric2:
            st.metric(
                "Risk Level",
                result["risk_level"]
            )

        with metric3:
            prediction_text = (
                "YES"
                if result["high_utilization_prediction"] == 1
                else "NO"
            )

            st.metric(
                "Predicted High Utilization",
                prediction_text
            )

        probability = result["risk_probability"]

        st.progress(probability)

        if result["risk_level"] == "HIGH":
            st.error(
                "High future healthcare utilization risk detected."
            )

        elif result["risk_level"] == "MODERATE":
            st.warning(
                "Moderate future healthcare utilization risk detected."
            )

        else:
            st.success(
                "Low future healthcare utilization risk detected."
            )

        st.subheader("Why this prediction?")

        for driver in result.get("risk_drivers", []):
            direction_icon = (
                "⬆️"
                if driver["direction"] == "increases risk"
                else "⬇️"
            )

            st.write(
                f"{direction_icon} **{driver['feature']}** "
                f"— {driver['direction']}"
            )

    except requests.exceptions.RequestException:

        st.error(
            "ClinSight API is unavailable. "
            "Make sure the FastAPI server is running."
        )


# --------------------------------------------------
# RAG question answering
# --------------------------------------------------

st.divider()

st.header("Ask ClinSight AI")

st.write(
    "Ask a healthcare utilization question. "
    "ClinSight AI will answer using its local knowledge base."
)

rag_question = st.text_input(
    "Your question",
    placeholder="Why might a patient have high healthcare utilization?"
)

if st.button(
    "Ask Clinsight AI",
    use_container_width=True
):

    if not rag_question.strip():
        st.warning("Please enter a question.")

    else:
        try:

            rag_response = requests.post(
                RAG_URL,
                json={
                    "question": rag_question
                },
                timeout=30
            )

            rag_response.raise_for_status()

            rag_result = rag_response.json()

            st.subheader("ClinSight Answer")
            st.write(rag_result["answer"])

            sources = rag_result.get(
                "sources",
                []
            )

            if sources:
                st.subheader("Sources")

                for source in sources:
                    st.write(f"- {source}")

        except requests.exceptions.RequestException:

            st.error(
                "The ClinSight RAG service is unavailable. "
                "Make sure the FastAPI server is running."
            )


# --------------------------------------------------
# LangGraph Agent
# --------------------------------------------------

st.divider()

st.header("ClinSight Agent")

st.write(
    "Ask ClinSight AI a question. The LangGraph agent will decide "
    "whether to use the healthcare knowledge base or the ML risk model."
)

agent_query = st.text_input(
    "Agent question",
    placeholder="Why might a patient have high healthcare utilization?",
    key="agent_query"
)

if st.button(
    "Ask ClinSight Agent",
    use_container_width=True
):

    if not agent_query.strip():
        st.warning("Please enter an agent question.")

    else:

        agent_payload = {
            "query": agent_query,
            "age": age,
            "gender": gender,
            "race": race,
            "ethnicity": ethnicity,
            "hist_total_encounters": encounters,
            "hist_total_conditions": conditions,
            "hist_total_procedures": procedures,
            "hist_total_medications": medications
        }

        try:

            agent_response = requests.post(
                AGENT_URL,
                json=agent_payload,
                timeout=30
            )

            agent_response.raise_for_status()

            agent_result = agent_response.json()

            st.subheader("Agent Response")

            st.write(
                f"**Route selected:** {agent_result['route']}"
            )

            st.write(
                agent_result["answer"]
            )

            if agent_result.get("source"):
                st.write(
                    f"**Source:** {agent_result['source']}"
                )

            if agent_result.get("risk_probability") is not None:

                st.metric(
                    "Agent Risk Probability",
                    f"{agent_result['risk_probability'] * 100:.2f}%"
                )

                st.write(
                    f"**Risk Level:** {agent_result['risk_level']}"
                )

        except requests.exceptions.RequestException:

            st.error(
                "ClinSight Agent is unavailable. "
                "Make sure the FastAPI server is running."
            )


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.divider()

st.caption(
    "ClinSight AI is a portfolio demonstration using synthetic "
    "healthcare data and is not intended for clinical decision-making."
)