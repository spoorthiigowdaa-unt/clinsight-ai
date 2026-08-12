import os

import requests
import streamlit as st


# --------------------------------------------------
# API configuration
# --------------------------------------------------

API_BASE_URL = os.getenv(
    "CLINSIGHT_API_URL",
    "http://127.0.0.1:8000"
)

API_URL = f"{API_BASE_URL}/predict"
RAG_URL = f"{API_BASE_URL}/rag/query"
AGENT_URL = f"{API_BASE_URL}/agent/query"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ClinSight AI",
    page_icon="🏥",
    layout="wide"
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1 {
        font-weight: 700;
    }

    h2, h3 {
        margin-top: 1.2rem;
    }

    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 16px;
        border-radius: 12px;
    }

    .section-card {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 14px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🏥 ClinSight AI")

st.caption(
    "Agentic Healthcare Intelligence Platform"
)

st.write(
    "Predict future healthcare utilization risk, explain model decisions, "
    "and answer grounded healthcare-utilization questions using ML, SHAP, "
    "RAG, and LangGraph."
)

st.divider()


# --------------------------------------------------
# 1. Patient demographics
# --------------------------------------------------

st.header("1. Patient Profile")

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
# 2. Historical utilization
# --------------------------------------------------

st.header(
    "2. Historical Healthcare Activity"
)

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
# 3. Prediction
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

        st.header(
            "3. Risk Assessment"
        )

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
                if result[
                    "high_utilization_prediction"
                ] == 1
                else "NO"
            )

            st.metric(
                "Predicted High Utilization",
                prediction_text
            )

        probability = result[
            "risk_probability"
        ]

        st.progress(
            probability
        )

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

        st.subheader(
            "Why this prediction?"
        )

        for driver in result.get(
            "risk_drivers",
            []
        ):

            direction_icon = (
                "⬆️"
                if driver["direction"]
                == "increases risk"
                else "⬇️"
            )

            st.write(
                f"{direction_icon} "
                f"**{driver['feature']}** "
                f"— {driver['direction']}"
            )

    except requests.exceptions.RequestException as exc:

        st.error(
            "ClinSight API is unavailable. "
            "Make sure the FastAPI server is running."
        )

        st.caption(
            str(exc)
        )


# --------------------------------------------------
# 4. RAG question answering
# --------------------------------------------------

st.divider()

st.header(
    "4. Ask ClinSight AI"
)

st.write(
    "Ask a healthcare utilization question. "
    "ClinSight AI will answer using its local knowledge base."
)

rag_question = st.text_input(
    "Your question",
    placeholder=(
        "Why might a patient have high healthcare utilization?"
    )
)

if st.button(
    "Ask ClinSight AI",
    use_container_width=True
):

    if not rag_question.strip():

        st.warning(
            "Please enter a question."
        )

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

            rag_result = (
                rag_response.json()
            )

            st.subheader(
                "ClinSight Answer"
            )

            st.write(
                rag_result["answer"]
            )

            sources = rag_result.get(
                "sources",
                []
            )

            if sources:

                st.subheader(
                    "Sources"
                )

                for source in sources:

                    st.write(
                        f"- {source}"
                    )

        except requests.exceptions.RequestException as exc:

            st.error(
                "The ClinSight RAG service is unavailable. "
                "Make sure the FastAPI server is running."
            )

            st.caption(
                str(exc)
            )


# --------------------------------------------------
# 5. LangGraph Agent
# --------------------------------------------------

st.divider()

st.header(
    "5. ClinSight Agent"
)

st.write(
    "Ask ClinSight AI a question. The LangGraph agent will decide "
    "whether to use the healthcare knowledge base, ML risk model, "
    "or explainability engine."
)

agent_query = st.text_input(
    "Agent question",
    placeholder=(
        "Explain why this patient has this risk score."
    ),
    key="agent_query"
)

if st.button(
    "Ask ClinSight Agent",
    use_container_width=True
):

    if not agent_query.strip():

        st.warning(
            "Please enter an agent question."
        )

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

            agent_result = (
                agent_response.json()
            )

            # ------------------------------------------
            # Agent response
            # ------------------------------------------

            st.subheader(
                "Agent Response"
            )

            route = agent_result.get(
                "route",
                "unknown"
            )

            st.info(
                f"Agent route selected: "
                f"{route.upper()}"
            )

            st.write(
                agent_result["answer"]
            )

            # ------------------------------------------
            # Source
            # ------------------------------------------

            if agent_result.get(
                "source"
            ):

                st.write(
                    f"**Source:** "
                    f"{agent_result['source']}"
                )

            # ------------------------------------------
            # Risk information
            # ------------------------------------------

            risk_probability = (
                agent_result.get(
                    "risk_probability"
                )
            )

            if risk_probability is not None:

                st.metric(
                    "Agent Risk Probability",
                    f"{risk_probability * 100:.2f}%"
                )

                st.write(
                    f"**Risk Level:** "
                    f"{agent_result['risk_level']}"
                )

            # ------------------------------------------
            # SHAP explanation
            # ------------------------------------------

            risk_drivers = (
                agent_result.get(
                    "risk_drivers",
                    []
                )
            )

            if risk_drivers:

                st.subheader(
                    "Why this prediction?"
                )

                for driver in risk_drivers:

                    direction_icon = (
                        "⬆️"
                        if driver["direction"]
                        == "increases risk"
                        else "⬇️"
                    )

                    st.write(
                        f"{direction_icon} "
                        f"**{driver['feature']}** "
                        f"— {driver['direction']}"
                    )

        except requests.exceptions.RequestException as exc:

            st.error(
                "ClinSight Agent is unavailable. "
                "Make sure the FastAPI server is running."
            )

            st.caption(
                str(exc)
            )


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.divider()

st.caption(
    "ClinSight AI uses synthetic healthcare data and is intended for "
    "educational and portfolio demonstration purposes only. "
    "It is not intended for clinical diagnosis, treatment, "
    "or decision-making."
)