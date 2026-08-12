# 🏥 ClinSight AI

## Agentic Healthcare Intelligence Platform

**Predictive ML • Explainable AI • RAG • Agentic AI • Healthcare Safety**

Clinsight AI is a portfolio-scale healthcare AI platform that combines **machine learning, explainable AI, retrieval-augmented generation (RAG), agentic AI, safety guardrails, FastAPI, and Streamlit**.

The system analyzes synthetic patient history to estimate future healthcare utilization risk, explains predictions using SHAP, answers healthcare utilization questions using a grounded local RAG pipeline, and uses a LangGraph workflow to route requests between the ML and knowledge-retrieval systems.

> **Disclaimer:** Clinsight AI is built using synthetic healthcare data and is intended for educational and portfolio demonstration purposes only. It is not intended for diagnosis, treatment recommendations, or clinical decision-making.

---

## 🚀 Key Features

* Healthcare utilization risk prediction
* Temporal feature engineering to reduce future-data leakage
* Logistic Regression, Random Forest, and Gradient Boosting model comparison
* SHAP-based patient-level explainability
* FastAPI REST API
* Interactive Streamlit dashboard
* Local RAG knowledge system
* SentenceTransformer embeddings
* ChromaDB vector database
* Local Hugging Face FLAN-T5 generation
* LangGraph agent orchestration
* ML vs. RAG request routing
* Healthcare safety guardrails
* Automated agent evaluation
* Docker and Docker Compose configuration

---

# 🧠 System Architecture

```text
                         Clinsight AI
                              │
                       Streamlit Dashboard
                              │
                           FastAPI
                              │
                       LangGraph Agent
                              │
                       Safety Guardrail
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          ML Prediction Route        RAG Knowledge Route
                 │                         │
          Random Forest                  Query
                 │                         │
          Scikit-learn               SentenceTransformer
                 │                         │
              SHAP                    Embedding
                 │                         │
       Risk + Explanation              ChromaDB
                                           │
                                     Relevant Chunks
                                           │
                                       FLAN-T5
                                           │
                                   Grounded Response
```

---

# 📊 Dataset

Clinsight AI uses **Synthea synthetic healthcare data**.

The project integrates six related healthcare datasets:

| Dataset     | Records | Columns |
| ----------- | ------: | ------: |
| Patients    |   1,163 |      25 |
| Encounters  |  61,459 |      15 |
| Conditions  |  38,094 |       6 |
| Procedures  |  83,823 |       9 |
| Medications |  56,430 |      13 |
| Claims      | 117,889 |      31 |

The raw datasets are intentionally excluded from GitHub.

---

# 🔍 Exploratory Data Analysis

The EDA pipeline analyzes:

* Patient demographics
* Age distribution
* Gender distribution
* Race and ethnicity
* Healthcare expenses
* Encounter utilization
* Medical conditions
* Procedures
* Medication activity
* Claims activity
* Missing values
* Duplicate records

No duplicate rows were detected across the six source datasets during the initial quality assessment.

The primary analysis notebook is:

```text
notebooks/01_healthcare_data_exploration.ipynb
```

---

# ⚙️ Feature Engineering

Raw event-level healthcare records are aggregated into patient-level machine-learning features.

Historical features include:

```text
AGE
GENDER
RACE
ETHNICITY
HIST_TOTAL_ENCOUNTERS
HIST_TOTAL_CONDITIONS
HIST_TOTAL_PROCEDURES
HIST_TOTAL_MEDICATIONS
```

The final modeling cohort contains:

```text
1,163 patients × 17 columns
```

---

# ⏳ Temporal Prediction Design

Clinsight AI uses a time-aware modeling approach.

Instead of using a patient's full history to predict an outcome derived from the same period, healthcare records are divided into:

```text
HISTORICAL PERIOD
       │
       ├── Encounters
       ├── Conditions
       ├── Procedures
       └── Medications
       │
       ▼
 Machine Learning Features

────────── Prediction Cutoff ──────────

       ▼
 FUTURE PERIOD
       │
       ▼
 Future Healthcare Utilization
```

This helps reduce **data leakage**, where future information could otherwise influence model training.

---

# 🎯 Prediction Target

The model predicts whether a patient will become a **high future healthcare utilizer**.

The high-utilization threshold was defined using the upper quartile of future encounter activity.

```text
Threshold: > 12 future encounters
```

Class distribution:

```text
Lower utilization     896 patients   77.04%
High utilization      267 patients   22.96%
```

---

# 🤖 Machine Learning Models

Three classification models were evaluated.

| Model               |  Accuracy | Precision | Recall |        F1 |   ROC-AUC |
| ------------------- | --------: | --------: | -----: | --------: | --------: |
| Logistic Regression |     0.670 |     0.367 |  0.623 |     0.462 |     0.690 |
| Random Forest       |     0.755 |     0.468 |  0.547 | **0.504** | **0.784** |
| Gradient Boosting   | **0.794** | **0.558** |  0.453 |     0.500 |     0.781 |

### Selected Model

**Random Forest**

Random Forest was selected as the current primary model because it provided the strongest overall balance of:

* ROC-AUC
* F1 score
* Recall
* Nonlinear modeling capability

---

# 🔧 Hyperparameter Tuning

Random Forest tuning was performed using:

```text
RandomizedSearchCV
5-fold cross-validation
ROC-AUC optimization
```

Parameters evaluated included:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

Tuning slightly increased recall but did not improve overall held-out performance.

| Model                    | Accuracy | Precision | Recall |    F1 | ROC-AUC |
| ------------------------ | -------: | --------: | -----: | ----: | ------: |
| Random Forest — Baseline |    0.755 |     0.468 |  0.547 | 0.504 |   0.784 |
| Random Forest — Tuned    |    0.734 |     0.435 |  0.566 | 0.492 |   0.783 |

The baseline Random Forest was therefore retained.

---

# 🔎 Explainable AI with SHAP

Clinsight AI uses **SHAP** to explain patient-specific predictions.

Instead of returning only:

```text
Risk Probability: 41.05%
Risk Level: MODERATE
```

the system also returns model-derived drivers such as:

```text
Historical Procedures      ↓ decreases risk
Historical Encounters      ↓ decreases risk
Gender: Female             ↑ increases risk
Historical Conditions      ↑ increases risk
Ethnicity: Non-Hispanic    ↓ decreases risk
```

This provides a more interpretable prediction experience.

> SHAP values explain model behavior and should not be interpreted as clinical causality.

---

# 💾 Model Persistence

The selected ML pipeline is serialized using **Joblib**:

```text
models/careguard_random_forest.joblib
```

The saved pipeline contains both preprocessing and the trained classifier so the API can perform inference without retraining.

---

# ⚡ FastAPI Backend

Clinsight AIexposes ML, RAG, and agent capabilities through FastAPI.

Current endpoints:

```text
GET   /
GET   /health

POST  /predict
POST  /rag/query
POST  /agent/query
```

Interactive Swagger documentation is available locally at:

```text
http://127.0.0.1:8000/docs
```

---

# 📈 Example ML Prediction

Example patient input:

```json
{
  "age": 58,
  "gender": "F",
  "race": "white",
  "ethnicity": "nonhispanic",
  "hist_total_encounters": 35,
  "hist_total_conditions": 12,
  "hist_total_procedures": 28,
  "hist_total_medications": 10
}
```

Example model result:

```text
Risk Probability: 41.05%
Risk Level: MODERATE
Predicted High Utilization: NO
```

---

# 🖥️ Streamlit Dashboard

The Streamlit application provides an interactive interface for:

### Patient Risk Assessment

Users enter:

* Age
* Gender
* Race
* Ethnicity
* Historical encounters
* Historical conditions
* Historical procedures
* Historical medications

Clinsight AIreturns:

* Risk probability
* Risk level
* High-utilization prediction
* SHAP risk drivers

### Ask Clinsight AI

Users can submit healthcare-utilization questions to the RAG system.

### Clinsight AIAgent

Users can interact with the LangGraph-powered agent, which determines which system should handle the request.

---

# 📚 Retrieval-Augmented Generation

Clinsight AIincludes a local RAG pipeline.

```text
Healthcare Knowledge Documents
              │
              ▼
      Document Ingestion
              │
              ▼
       Text Chunking
              │
              ▼
 SentenceTransformer Embeddings
              │
              ▼
          ChromaDB
              │
              ▼
      Semantic Retrieval
              │
              ▼
       Local FLAN-T5
              │
              ▼
  Grounded Answer + Source
```

---

# 🧩 RAG Components

### Knowledge Base

```text
data/knowledge_base/
```

### Document Ingestion

```text
src/rag/ingest.py
```

Documents are loaded and split using recursive text chunking.

Current configuration:

```text
Chunk size:     500
Chunk overlap:  100
```

### Embeddings

Embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Vector Database

```text
ChromaDB
```

### Local Generator

```text
google/flan-t5-small
```

The generation layer runs locally and does not require paid OpenAI API credits.

---

# 🔎 Semantic Retrieval Example

Question:

```text
Why might a patient need a lot of medical care?
```

The vector search retrieves relevant knowledge even when the wording does not exactly match the source document.

This demonstrates **semantic retrieval rather than simple keyword matching**.

---

# 🧠 LangGraph Agent

Clinsight AIuses **LangGraph** to orchestrate different AI capabilities.

The agent evaluates each user request and routes it to the appropriate system.

```text
User Request
      │
      ▼
Safety Check
      │
      ▼
LangGraph Router
      │
 ┌────┴─────────────┐
 │                  │
 ▼                  ▼
RAG Route      Prediction Route
 │                  │
ChromaDB       Random Forest
+ FLAN-T5      Risk Model
 │                  │
 └────────┬─────────┘
          ▼
   Unified Response
```

Example:

```text
Question:
Why might a patient have high healthcare utilization?

Route:
rag
```

Example:

```text
Question:
Predict this patient's healthcare utilization risk.

Route:
prediction

Result:
41.05% — MODERATE risk
```

---

# 🛡️ Safety Guardrails

The LangGraph workflow includes a safety layer that blocks requests for:

* Diagnosis
* Prescriptions
* Medication selection
* Medication changes
* Dosage recommendations
* Treatment instructions

Example request:

```text
What medication should I take for high healthcare utilization?
```

Clinsight AIresponse:

```text
Clinsight AI does not provide diagnosis, prescriptions,
medication changes, or treatment advice.
Please consult a qualified healthcare professional.
```

The workflow routes these requests to:

```text
Clinsight AISafety Guardrail
```

instead of the RAG or ML systems.

---

# ✅ Agent Evaluation

A custom evaluation suite tests:

* Routing behavior
* Safety blocking
* RAG source grounding
* Prediction response structure

Current evaluation:

```text
Routing tests:       3/3 passed
Safety tests:        4/4 passed
RAG grounding:       1/1 passed
Prediction test:     1/1 passed

Total:               9/9 passed
Evaluation score:    100%
```

> The 100% score refers only to the project's current 9-case custom integration evaluation suite. It does not represent 100% ML accuracy, clinical reliability, or universal safety performance.

Evaluation code:

```text
src/evaluation/agent.py
```

---

# 🧰 Technology Stack

### Data & Machine Learning

```text
Python
Pandas
NumPy
Scikit-learn
Joblib
```

### Machine Learning Models

```text
Logistic Regression
Random Forest
Gradient Boosting
```

### Explainable AI

```text
SHAP
```

### Generative AI / RAG

```text
Hugging Face Transformers
FLAN-T5
SentenceTransformers
ChromaDB
LangChain Text Splitters
```

### Agentic AI

```text
LangGraph
```

### Backend

```text
FastAPI
Pydantic
Uvicorn
```

### Frontend

```text
Streamlit
Requests
```

### Engineering

```text
Git
GitHub
Docker
Docker Compose
Jupyter Notebook
VS Code
```

---

# 📁 Project Structure

```text
careguard-ai/
│
├── data/
│   ├── knowledge_base/
│   │   └── healthcare_utilization.md
│   │
│   ├── raw/                     # ignored by Git
│   ├── processed/
│   ├── synthetic/
│   └── chroma_db/               # generated locally
│
├── frontend/
│   └── app.py
│
├── models/
│   └── careguard_random_forest.joblib
│
├── notebooks/
│   └── 01_healthcare_data_exploration.ipynb
│
├── src/
│   ├── agents/
│   │   └── careguard_agent.py
│   │
│   ├── api/
│   │   └── main.py
│   │
│   ├── evaluation/
│   │   └── agent.py
│   │
│   ├── features/
│   ├── ingestion/
│   ├── models/
│   ├── preprocessing/
│   │
│   └── rag/
│       ├── ingest.py
│       ├── retriever.py
│       └── generator.py
│
├── tests/
│
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 💻 Local Setup

## 1. Clone the repository

```bash
git clone https://github.com/spoorthiigowdaa-unt/careguard-ai.git
cd careguard-ai
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📚 Build the RAG Vector Store

Run:

```bash
python -m src.rag.retriever
```

This creates the local ChromaDB vector store from documents stored in:

```text
data/knowledge_base/
```

---

# ⚡ Start FastAPI

```bash
python -m uvicorn src.api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

# 🖥️ Start Streamlit

In another terminal:

```bash
python -m streamlit run frontend/app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🧪 Run Agent Evaluation

```bash
python -m src.evaluation.agent
```

Expected current result:

```text
Passed: 9/9
Evaluation Score: 100.0%
```

---

# 🐳 Docker

Docker configuration is included:

```text
Dockerfile
compose.yaml
.dockerignore
```

The intended command is:

```bash
docker compose up --build
```

The Compose architecture runs:

```text
careguard-api        → FastAPI     → port 8000
careguard-dashboard  → Streamlit   → port 8501
```

> Docker configuration has been added to the repository. Full local container validation is still pending.

---

# 🔐 Security

Sensitive and generated local files are excluded from Git.

Examples include:

```text
.env
.venv/
data/raw/
data/chroma_db/
```

Never commit API keys, credentials, real patient data, or local environment files.

---

# 🗺️ Roadmap

Future improvements include:

* Expanded healthcare knowledge base
* Stronger local instruction model
* Advanced RAG evaluation
* Retrieval relevance scoring
* Citation-level grounding evaluation
* Claims anomaly detection
* FHIR R4 ingestion and analytics
* FHIR-aware RAG
* MLflow experiment tracking
* Model monitoring and drift detection
* Redis-based caching
* PostgreSQL persistence
* Authentication and RBAC
* Expanded LangGraph workflows
* Human-in-the-loop review
* Docker deployment validation
* Cloud deployment
* CI/CD with GitHub Actions
* Automated unit/integration testing
* Healthcare fairness and subgroup evaluation

---

# 🎯 Project Goal

Clinsight AI demonstrates how multiple AI/ML technologies can be combined into one end-to-end healthcare intelligence system:

```text
Healthcare Data
      ↓
Feature Engineering
      ↓
Temporal ML Prediction
      ↓
Explainable AI
      ↓
FastAPI
      ↓
Streamlit
      ↓
RAG
      ↓
LangGraph Agent
      ↓
Safety Guardrails
      ↓
Evaluation
```

The project is designed to demonstrate practical skills across **machine learning, generative AI, RAG, agentic AI, explainability, backend engineering, and AI system design**.

---

## 👩‍💻 Author

**Spoorthi Hassan Sathyanarayana**

GitHub: [spoorthiigowdaa-unt](https://github.com/spoorthiigowdaa-unt)

---

## ⚠️ Responsible AI Disclaimer

Clinsight AI uses synthetic data and is intended solely for educational, research, and portfolio demonstration purposes.

The system does not provide medical diagnosis, prescriptions, treatment recommendations, or clinical decision support. Predictions and generated responses should not be used for real-world healthcare decisions.
