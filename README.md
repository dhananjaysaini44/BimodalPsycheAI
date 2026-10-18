# BimodalPsycheAI — Multimodal Depression Detection System

**BimodalPsycheAI** is an adaptive, conversational mental health screening system that evaluates depression risk using bimodal fusion across **acoustic vocal biomarkers** (prosodic & spectral features) and **conversational text dynamics** (aligned with clinical PHQ-8 and PHQ-9 assessment frameworks).

For visual workflows and data structures, see [`docs/diagrams/`](docs/diagrams/) for architecture blueprints, Level-0 and Level-1 Data Flow Diagrams (DFD), and sequence diagrams.

---

## Tech Stack

### Languages & Runtimes
<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Java-21%20%7C%2023-ED8B00.svg?style=flat&logo=openjdk&logoColor=white" alt="Java" />
  <img src="https://img.shields.io/badge/TypeScript-5.x-3178C6.svg?style=flat&logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/JavaScript-ES2024-F7DF1E.svg?style=flat&logo=javascript&logoColor=black" alt="JavaScript" />
  <img src="https://img.shields.io/badge/SQL-MySQL%20%26%20ANSI%20SQL-4479A1.svg?style=flat&logo=mysql&logoColor=white" alt="SQL" />
</p>

### Backend & API Gateway
<p align="left">
  <img src="https://img.shields.io/badge/Spring_Boot-3.x-6DB33F.svg?style=flat&logo=springboot&logoColor=white" alt="Spring Boot" />
  <img src="https://img.shields.io/badge/Spring_Security-6.x-6DB33F.svg?style=flat&logo=springsecurity&logoColor=white" alt="Spring Security" />
  <img src="https://img.shields.io/badge/Spring_Data_JPA-Hibernate-59666C.svg?style=flat&logo=hibernate&logoColor=white" alt="Spring Data JPA" />
  <img src="https://img.shields.io/badge/JWT-JJWT_0.11.5-000000.svg?style=flat&logo=jsonwebtokens&logoColor=white" alt="JWT" />
  <img src="https://img.shields.io/badge/Apache_Maven-3.9+-C71A36.svg?style=flat&logo=apachemaven&logoColor=white" alt="Maven" />
  <img src="https://img.shields.io/badge/Project_Lombok-Boilerplate-BC4522.svg?style=flat&logo=oracle&logoColor=white" alt="Lombok" />
</p>

### Machine Learning, Audio DSP & NLP
<p align="left">
  <img src="https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?style=flat&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Uvicorn-ASGI_Server-499848.svg?style=flat&logo=gunicorn&logoColor=white" alt="Uvicorn" />
  <img src="https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.5.0-F7931E.svg?style=flat&logo=scikitlearn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Hugging_Face-Transformers_4.42-FFD21E.svg?style=flat&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Librosa-0.10.2_Audio_DSP-2B5B84.svg?style=flat&logo=soundcharts&logoColor=white" alt="Librosa" />
  <img src="https://img.shields.io/badge/openSMILE-eGeMAPS_Acoustics-4B8BBE.svg?style=flat&logo=audioboom&logoColor=white" alt="openSMILE" />
  <img src="https://img.shields.io/badge/SoundFile-0.12.1-00599C.svg?style=flat&logo=waveform&logoColor=white" alt="SoundFile" />
  <img src="https://img.shields.io/badge/VADER-Sentiment_Analysis-4B8BBE.svg?style=flat&logo=nltk&logoColor=white" alt="VADER" />
  <img src="https://img.shields.io/badge/Pandas-2.2.2-150458.svg?style=flat&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-1.26.4-013243.svg?style=flat&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Matplotlib-3.9.0-11557C.svg?style=flat&logo=python&logoColor=white" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg?style=flat&logo=jupyter&logoColor=white" alt="Jupyter" />
</p>

### Frontend & Client UI
<p align="left">
  <img src="https://img.shields.io/badge/Next.js-14%2F15-000000.svg?style=flat&logo=nextdotjs&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/React-18.x-61DAFB.svg?style=flat&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/HTML5-Web_Audio_API-E34F26.svg?style=flat&logo=html5&logoColor=white" alt="HTML5" />
  <img src="https://img.shields.io/badge/CSS3-Modern_Styling-1572B6.svg?style=flat&logo=css3&logoColor=white" alt="CSS3" />
</p>

### Databases, Infrastructure & DevOps
<p align="left">
  <img src="https://img.shields.io/badge/MySQL-8.x-4479A1.svg?style=flat&logo=mysql&logoColor=white" alt="MySQL" />
  <img src="https://img.shields.io/badge/H2-In_Memory_Testing-003B57.svg?style=flat&logo=database&logoColor=white" alt="H2 Database" />
  <img src="https://img.shields.io/badge/Docker-Containerization-2496ED.svg?style=flat&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Docker_Compose-Multi--Container-2496ED.svg?style=flat&logo=docker&logoColor=white" alt="Docker Compose" />
  <img src="https://img.shields.io/badge/Git-Version_Control-F05032.svg?style=flat&logo=git&logoColor=white" alt="Git" />
  <img src="https://img.shields.io/badge/GitHub-Repo-181717.svg?style=flat&logo=github&logoColor=white" alt="GitHub" />
  <img src="https://img.shields.io/badge/Mermaid.js-Architecture_Diagrams-FF3670.svg?style=flat&logo=mermaid&logoColor=white" alt="Mermaid" />
</p>

---

## System Architecture

The project is structured into modular microservices and pipelines:

* **`backend/` (API Gateway & Core Business Logic)**: Built with Spring Boot 3 and Java 23. Manages user registration, email OTP verification, JWT session authentication, account lifecycle, and clinical report queries.
* **`frontend/` (Conversational UI)**: Built with Next.js and TypeScript. Captures real-time audio through the browser Web Audio API alongside text responses for conversational assessment.
* **`ml-engine/` (Model Serving & Orchestration)**: Built with FastAPI and Python. Provides low-latency HTTP inference for turn-by-turn sentiment scoring, audio feature evaluation, rule-based crisis detection, and decision fusion.
* **`ml-training/` (Offline ML Research & Training Pipeline)**: Python research environment that processes the E-DAIC benchmark. Contains feature extraction pipelines (MFCC, eGeMAPS, BoAW), temporal pooling (mean, std, min, max), repeated cross-validation protocols, and baseline models (LinearSVC, Logistic Regression, Ridge, Random Forest).
* **`docs/` (Architecture & Specifications)**: Detailed system synopsis, architecture diagrams (Mermaid & PNG), and data use agreements.

---

## Repository Structure

```
BimodalPsycheAI/
├── .gitignore                                 # Git ignore rules for datasets, model weights, and targets
├── docker-compose.yml                         # Orchestration for Backend and ML-Engine containers
├── README.md                                  # Comprehensive repository documentation
│
├── backend/                                   # Spring Boot 3 API Gateway (Java 23, Maven)
│   ├── mvnw / mvnw.cmd                        # Cross-platform Maven wrapper
│   ├── pom.xml                                # Spring Web, Security, Data JPA, MySQL, JJWT, Validation
│   └── src/
│       ├── main/java/com/bimodalpsyche/backend/
│       │   ├── BackendApplication.java        # Spring Boot entry point
│       │   ├── auth/                          # Auth controllers, JWT filter, OTP & refresh token services
│       │   ├── config/                        # Spring Security and CORS filter chains
│       │   ├── exception/                     # Global exception handlers and error response payloads
│       │   └── user/                          # User entity, enums (Role, Gender), and account repository
│       ├── main/resources/
│       │   └── application.properties         # Server port, MySQL connection, and mail properties
│       └── test/                              # Spring Boot integration & unit test suite
│
├── frontend/                                  # Next.js Conversational UI (TypeScript)
│   └── .gitkeep                               # Client scaffolding placeholder
│
├── ml-engine/                                 # FastAPI Real-time Inference Microservice (Python 3.10+)
│   ├── requirements.txt                       # Serving dependencies: FastAPI, Uvicorn, Librosa, vaderSentiment
│   ├── app/
│   │   ├── main.py                            # FastAPI application entry point (port 8000)
│   │   ├── orchestrator/
│   │   │   ├── crisis_detector.py             # Rule-based crisis keywords and risk evaluation
│   │   │   └── sentiment_scorer.py            # VADER turn-by-turn conversational sentiment scoring
│   │   └── routers/
│   │       ├── fusion.py                      # Multimodal decision & score fusion
│   │       ├── orchestrator.py                # Adaptive assessment flow state controller
│   │       ├── text_analysis.py               # NLP text feature extraction
│   │       └── voice_analysis.py              # Acoustic audio evaluation
│   ├── data/
│   │   └── question_bank.json                 # Structured assessment questions (intro, PHQ-8/9, wrap-up)
│   └── models/
│       └── README.md                          # Production weight drop directory (git-ignored)
│
├── ml-training/                               # Offline ML Pipeline & Benchmark Research Environment
│   ├── requirements.txt                       # Training dependencies: scikit-learn, pandas, numpy, librosa, torch
│   ├── methodology.md                         # Benchmark experimental protocol and methodology
│   ├── model_selection.md                     # Model evaluation metrics, CV analysis, and model selection
│   ├── BiModelPsycheAI Starting Checklist.md  # Training pipeline checklist and action items
│   ├── edaic-woz/                             # [GIT-IGNORED] USC ICT Private Dataset & Labels (241+ GB)
│   │   ├── data/                              # Raw audio (.wav), interview transcripts, openSMILE CSVs
│   │   ├── labels/                            # Official train, dev, and test clinical split CSVs
│   │   ├── labels2019/                        # AVEC challenge label archives
│   │   ├── metadata_mapped.csv                # Participant identifier mapping table
│   │   └── E-DAIC Manual.pdf                  # Official dataset specification and protocol manual
│   ├── dataset/                               # [GIT-IGNORED] Generated feature tables (mean, std, BoAW)
│   ├── datasets/
│   │   └── README.md                          # Access guidelines for clinical datasets
│   ├── outputs/                               # [GIT-IGNORED] Repeated-CV logs, ROC/PR curves, predictions
│   └── code/                                  # Modular ML pipeline source code (Version Controlled)
│       ├── config.py                          # Dynamic path auto-detection for datasets and outputs
│       ├── dataset_generation.py              # Feature pooling & master dataset assembler
│       ├── feature_extraction.py              # Acoustic Low-Level Descriptor (LLD) parsers
│       ├── pooling.py                         # Temporal pooling primitives (mean, std, min, max)
│       ├── split_loader.py                    # Official train/dev/test split loaders & CV partitioners
│       ├── run_repeated_cv.py                 # Repeated cross-validation stability protocol
│       ├── summarize_models.py                # Performance summary and LaTeX/Markdown reporter
│       ├── evaluation/                        # Classification & regression evaluators
│       ├── experiments/                       # Experiment runners and model architecture registry
│       ├── models/                            # Classification and regression model architectures
│       ├── utils/                             # Metrics (CCC, RMSE, MAE, F1, Balanced Accuracy)
│       ├── visualization/                     # Heatmap, confusion matrix, ROC, and PR curve generation
│       ├── Baseline_models.ipynb              # Baseline models exploratory notebook
│       ├── Validation_data.ipynb              # Validation data pipeline notebook
│       └── test_*.py                          # Complete 33-test unit test suite
│
└── docs/                                      # System Specifications & Schemas
    ├── data-use-agreement/                    # Clinical dataset compliance notes
    ├── diagrams/                              # Architecture diagrams (Mermaid sources + rendered PNGs)
    └── synopsis/                              # Project synopsis, clinical motivation, and objectives
```

---

## Detailed Getting Started Guide

### 1. Prerequisites
* **Java**: JDK 21 or JDK 23 installed (`java -version`)
* **Python**: Python 3.10, 3.11, or 3.12 with `pip`
* **MySQL**: MySQL Server 8.x running locally on port 3306 (or configured via environment)
* **Node.js**: Node 18+ and `npm` (for the frontend client)
* **Docker & Docker Compose** (Optional, for containerized deployments)

---

### 2. Backend Service (Spring Boot 3)

The backend handles account security, email OTP verification, JWT authentication, and session coordination.

#### Configuration
Set the required environment variables or update `backend/src/main/resources/application.properties`:
* `JWT_SECRET`: A secure base64-encoded HMAC-SHA secret (at least 256 bits).
* `MAIL_USERNAME`: Gmail/SMTP account email for OTP dispatch.
* `MAIL_PASSWORD`: Gmail App Password for SMTP authentication.

```bash
# Navigate to the backend directory
cd backend

# Compile and run unit tests using the Maven wrapper
# Linux / macOS:
./mvnw clean test-compile
# Windows (PowerShell or CMD):
.\mvnw.cmd clean test-compile

# Start the Spring Boot backend server (default port: 8080 or configured port)
# Linux / macOS:
./mvnw spring-boot:run
# Windows:
.\mvnw.cmd spring-boot:run
```

---

### 3. ML Serving Engine (FastAPI)

The ML Engine exposes high-performance asynchronous REST endpoints for real-time feature evaluation and adaptive assessment coordination.

```bash
# Navigate to ml-engine
cd ml-engine

# Create and activate a Python virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Launch the FastAPI application on port 8000
uvicorn app.main:app --reload --port 8000
```

Once running, interactive documentation is available at:
* Swagger UI: `http://localhost:8000/docs`
* ReDoc UI: `http://localhost:8000/redoc`

---

### 4. ML Training & Benchmark Pipeline (Offline Research)

The training pipeline supports acoustic feature extraction, temporal pooling, cross-validation model benchmarking, and full regression/classification stability audits.

```bash
# Navigate to ml-training
cd ml-training

# Install ML research dependencies
pip install -r requirements.txt

# Run the complete test suite (33 tests covering split loaders, feature extractors, pooling, and metrics)
python -m unittest discover -s code -p "test_*.py"

# Generate pooled feature datasets from raw features
python code/dataset_generation.py

# Run classification benchmark across all feature representations
python code/experiments/run_classification.py

# Run regression benchmark targeting continuous PHQ scores
python code/experiments/run_regression.py

# Run 10-repeat 5-fold cross-validation stability protocol
python code/run_repeated_cv.py
```

---

### 5. Frontend Client (Next.js)

```bash
# Navigate to the frontend directory
cd frontend

# Install client dependencies
npm install

# Start the local development server (http://localhost:3000)
npm run dev
```

---

### 6. Full Stack Deployment via Docker Compose

To build and spin up the backend API gateway and ML serving engine simultaneously:

```bash
docker-compose up --build
```

---

## Dataset Notice & Ethics Compliance

This repository utilizes experimental data from the **Extended Distress Analysis Interview Corpus (E-DAIC / DAIC-WOZ)** distributed by the University of Southern California (USC) Institute for Creative Technologies.

* **Data-Use Agreement Compliance**: In strict adherence to the USC ICT data-use agreement, raw audio files, interview transcripts, and clinical PHQ-8/PHQ-9 diagnostic labels are **strictly Git-ignored** and must never be committed to public or private version control.
* **Storage Location**: Authorized researchers should place their approved dataset within `ml-training/edaic-woz/data/` and `ml-training/edaic-woz/labels/`. All pipeline scripts dynamically detect and consume files from this directory automatically.

---

### References & Citations

If you produce a paper or project that in any way utilizes this database, please remember to cite the foundational references below:

1. **The Distress Analysis Interview Corpus (DAIC)**:
   > Gratch J, Artstein R, Lucas GM, Stratou G, Scherer S, Nazarian A, Wood R, Boberg J, DeVault D, Marsella S, Traum DR.  
   > *The Distress Analysis Interview Corpus of human and computer interviews.*  
   > In LREC 2014 May (pp. 3123–3128).  
   > [[Paper Link](http://www.lrec-conf.org/proceedings/lrec2014/pdf/508_Paper.pdf)]

2. **AVEC 2019 Depression Challenge (E-DAIC)**:
   > Ringeval, Fabien, Björn Schuller, Michel Valstar, Nicholas Cummins, Roddy Cowie, Leili Tavabi, Maximilian Schmitt et al.  
   > *"AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition."*  
   > In Proceedings of the 9th International on Audio/Visual Emotion Challenge and Workshop, pp. 3–12. ACM, 2019.  
   > [[ACM Digital Library](https://doi.org/10.1145/3347320.3357688)]
