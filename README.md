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
  <img src="https://img.shields.io/badge/Seaborn-0.13.2-4C72B0.svg?style=flat&logo=python&logoColor=white" alt="Seaborn" />
  <img src="https://img.shields.io/badge/Plotly-7.1.0-3F4F75.svg?style=flat&logo=plotly&logoColor=white" alt="Plotly" />
  <img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg?style=flat&logo=jupyter&logoColor=white" alt="Jupyter" />
</p>

### Deep Learning Architectures & Transformer Embeddings (ML Training)
<p align="left">
  <img src="https://img.shields.io/badge/CNN-VGG16_%26_DenseNet201-00599C.svg?style=flat&logo=pytorch&logoColor=white" alt="CNN" />
  <img src="https://img.shields.io/badge/GRU-Temporal_Sequence_Modeling-008080.svg?style=flat&logo=pytorch&logoColor=white" alt="GRU" />
  <img src="https://img.shields.io/badge/Attention-Additive_%26_Masked_Pooling-8A2BE2.svg?style=flat&logo=pytorch&logoColor=white" alt="Attention" />
  <img src="https://img.shields.io/badge/Transformers-Embedding_Extraction-FFA000.svg?style=flat&logo=huggingface&logoColor=black" alt="Transformers Embeddings" />
  <img src="https://img.shields.io/badge/MiniLM-all--MiniLM--L6--v2-FF6F00.svg?style=flat&logo=huggingface&logoColor=white" alt="MiniLM" />
  <img src="https://img.shields.io/badge/DistilBERT-distilbert--base--uncased-FF9900.svg?style=flat&logo=huggingface&logoColor=black" alt="DistilBERT" />
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
* **`ml-training/` (Offline ML Research & Training Pipeline)**: Python research environment that processes the E-DAIC benchmark. Contains feature extraction pipelines (MFCC, eGeMAPS, BoAW, deep CNN audio embeddings via VGG16/DenseNet201), temporal sequence modeling (GRU with masked attention), transcript chunk embeddings using Transformers (all-MiniLM-L6-v2, DistilBERT), additive attention pooling, repeated cross-validation protocols, diagnostic visualization (Matplotlib, Seaborn, Plotly), and benchmark models.
* **`docs/` (Architecture & Specifications)**: Detailed system synopsis, architecture diagrams (Mermaid & PNG), and data use agreements.

---

## Repository Structure

```
BimodalPsycheAI/
├── .github/                                   # GitHub workflow templates and modernization hooks
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
│   ├── requirements.txt                       # Training dependencies: scikit-learn, pandas, numpy, librosa, torch, transformers
│   ├── methodology.md                         # Benchmark experimental protocol, forensic audits, and methodology
│   ├── model_selection.md                     # Model evaluation metrics, CV analysis, and model selection
│   ├── BiModelPsycheAI Starting Checklist.md  # Training pipeline checklist and action items
│   ├── edaic-woz/                             # [GIT-IGNORED] USC ICT Private Dataset & Labels (241+ GB)
│   │   ├── data/                              # Raw audio (.wav), interview transcripts, openSMILE CSVs
│   │   ├── labels/                            # Official train, dev, and test clinical split CSVs
│   │   ├── labels2019/                        # AVEC challenge label archives
│   │   ├── metadata_mapped.csv                # Participant identifier mapping table
│   │   └── E-DAIC Manual.pdf                  # Official dataset specification and protocol manual
│   ├── dataset/                               # [GIT-IGNORED] Generated feature tables and split matrices
│   │   ├── raw_egemaps/ & raw_mfcc/           # 100 Hz low-level descriptors (eGeMAPS 23-d, MFCC 39-d)
│   │   ├── boaw_egemaps/ & boaw_mfcc/         # 10 Hz 100-bin Bag-of-Audio-Words quantized histograms
│   │   ├── deep_features_v1/                  # Pretrained CNN transfer audio representations (VGG16 & DenseNet201)
│   │   ├── text_v1/                           # Participant-level TF-IDF sparse matrices (Phase 17A)
│   │   ├── text_embeddings_v1/                # Participant-level MiniLM-L6-v2 dense embeddings (Phase 17B)
│   │   ├── text_segments_v1/                  # Chronological segment-level MiniLM embeddings (Phase 18A)
│   │   ├── text_distilbert_v1/                # Chunk-level & segment-aggregated DistilBERT embeddings (Phase 18B/18C)
│   │   └── splits/                            # Partition-aligned feature tables across Train, Dev, and Test
│   ├── datasets/
│   │   └── README.md                          # Access guidelines for clinical datasets
│   ├── outputs/                               # Benchmark logs, evaluation metrics, and predictions
│   │   ├── classical_controlled/              # Phase 9: Controlled benchmark outputs (mean_std protocol)
│   │   ├── classical_final_test_v1/           # Phase 10: Official Test release, 9 serialized models & manifest
│   │   ├── cnn_embedding_classical_v1/        # Stage 1 / Phase 13A: Classical models on CNN audio representations
│   │   ├── cnn_embedding_neural_head_v1/      # Stage 2 / Phase 13A: Deep MLP prediction heads on CNN embeddings
│   │   ├── cnn_embedding_temporal_v1/         # Stage 13B: GRU temporal sequence modeling with masked attention
│   │   ├── text_baselines_v1/                 # Phase 17A: Participant-level TF-IDF baseline models & evaluation
│   │   ├── text_baselines_final_v1/           # Phase 17A: Final model release on Train OOF & Development splits
│   │   ├── text_embeddings_v1/                # Phase 17B: Frozen all-MiniLM-L6-v2 text embeddings & models
│   │   ├── text_embeddings_tuned_v1/          # Phase 17B.1: Hyperparameter-tuned MiniLM models & grid records
│   │   ├── text_mlp_v1/                       # Phase 17C: PyTorch neural MLP on MiniLM embeddings
│   │   ├── text_segments_v1/                  # Phase 18A: Chronological segment embeddings (early/mid/late) & aggregation
│   │   ├── text_distilbert_v1/                # Phase 18B: DistilBERT chunk embeddings & multi-segment pipelines
│   │   ├── text_attention_v1/                 # Phase 18C: Additive attention pooling on DistilBERT chunk representations
│   │   ├── phase_18a_audit/                   # Phase 18A forensic audit verification report & artifacts
│   │   ├── phase_18b_audit/                   # Phase 18B forensic audit verification report & artifacts
│   │   ├── phase_18c_audit/                   # Phase 18C forensic audit verification report & artifacts
│   │   ├── model_visualizations_v1/           # Diagnostic visualizations (ROC/PR curves, scatter plots, attention maps)
│   │   ├── classification/                    # Historical classification exploration outputs & curves
│   │   └── regression/                        # Historical regression exploration outputs & curves
│   └── code/                                  # Modular ML pipeline source code (Version Controlled)
│       ├── core/                              # Central configurations and global project paths
│       │   └── config.py                      # Canonical paths, random seed, and feature specifications
│       ├── data/                              # Data loading, temporal pooling, and feature extraction
│       │   ├── split_loader.py                # Leakage-safe official split loader (Train=163, Dev=56, Test=56)
│       │   ├── pooling.py                     # Temporal pooling functions (mean, std, min, max, mean_std)
│       │   ├── feature_extraction.py          # Acoustic LLD and BoAW feature extraction routines
│       │   └── dataset_generation.py          # Generates 24 pooled representation datasets + manifests
│       ├── evaluation/                        # Model evaluators and validation splits
│       │   ├── classification_evaluator.py    # Stratified CV & Dev classification evaluator
│       │   ├── regression_evaluator.py        # Regression evaluator with CCC, RMSE, MAE, Pearson
│       │   └── cross_validation.py            # Cross-validation fold partitioners
│       ├── experiments/                       # Benchmark runners and experiment entry points
│       │   ├── run_classical_controlled_benchmark.py     # Phase 9: 4-representation controlled benchmark
│       │   ├── run_classical_final_test.py               # Phase 10: Official test evaluation & model serialization
│       │   ├── run_cnn_embedding_classical_benchmark.py  # Stage 1: Classical models on VGG16/DenseNet201
│       │   ├── run_cnn_embedding_neural_head_benchmark.py # Stage 2: MLP neural heads on CNN embeddings
│       │   ├── run_cnn_embedding_temporal_benchmark.py   # Stage 13B: GRU temporal sequence modeling
│       │   ├── run_repeated_cv.py                        # 50-fold repeated CV stability (classification)
│       │   ├── run_repeated_cv_regression.py             # 50-fold repeated CV stability (regression)
│       │   ├── run_classification.py                     # Broad classical classification exploration
│       │   └── run_regression.py                         # Broad classical regression exploration
│       ├── models/                            # Classical model architectures
│       │   ├── classification_models.py       # LinearSVC, LogisticRegression, RF, ExtraTrees, XGBoost
│       │   └── regression_models.py           # Ridge, Lasso, ElasticNet, SVR, KNN, RF, ExtraTrees, XGBoost
│       ├── text_baselines_v1/                 # Phase 17A: TF-IDF feature extraction, training & evaluation
│       ├── text_embeddings_v1/                # Phase 17B: MiniLM embedding extraction & baseline modeling
│       ├── text_embeddings_tuned_v1/          # Phase 17B.1: Hyperparameter tuning & grid evaluations
│       ├── text_mlp_v1/                       # Phase 17C: PyTorch neural MLP architectures & trainers
│       ├── text_segments_v1/                  # Phase 18A: Chronological segmentation & robust aggregation
│       ├── text_distilbert_v1/                # Phase 18B: DistilBERT chunk embedding & multi-segment pipelines
│       ├── text_attention_v1/                 # Phase 18C: Lightweight additive attention pooling & diagnostics
│       ├── audits/                            # Forensic validation scripts & audit test harnesses
│       │   ├── audit_classical_controlled.py  # 63-check verification of controlled benchmark
│       │   ├── audit_phases1_to_4.py          # Split integrity, metric verification & fold stability
│       │   ├── audit_regression_forensic.py   # Regression benchmark integrity & prediction bounds
│       │   └── audit_rf_anomaly.py            # Random Forest zero-positive probability diagnosis
│       ├── tests/                             # Automated test suite (18 suites covering audio DSP, CNN, GRU, text, attention)
│       ├── utils/                             # Shared metric implementations (Lin's CCC, F1, Balanced Acc, bootstrap tests)
│       ├── visualization/                     # ROC curves, PR curves, regression scatters, and attention visualizers
│       ├── notebooks/                         # Jupyter exploration and prototyping notebooks
│       ├── scripts/                           # Standalone utility and reporting scripts
│       └── [Compatibility Shims]              # Root re-exports (config.py, split_loader.py, pooling.py, etc.)
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

The training pipeline supports acoustic feature extraction (eGeMAPS, MFCC, BoAW, CNN transfer learning), temporal sequence modeling (GRU), transcript chunk embedding extraction using Transformers (MiniLM, DistilBERT), additive attention pooling, cross-validation model benchmarking, diagnostic visualization (Seaborn, Plotly, Matplotlib), and full regression/classification stability audits.


```bash
# Navigate to ml-training
cd ml-training

# Install ML research dependencies
pip install -r requirements.txt

# Run the complete test suite (18 test suites covering split loaders, feature extractors, CNN, GRU, Transformers, attention, and metrics)
python -m unittest discover -s code/tests -p "test_*.py"

# Run controlled classical benchmark audit (63 automated integrity checks)
python code/audits/audit_classical_controlled.py

# Generate pooled feature datasets from raw features
python code/data/dataset_generation.py

# Run the controlled classical benchmark (mean_std pooling across 4 acoustic representations)
python code/experiments/run_classical_controlled_benchmark.py

# Run the official test set evaluation & save fitted models (Phase 10)
python code/experiments/run_classical_final_test.py

# Summarize model benchmark performance across classification and regression
python code/scripts/summarize_models.py
```
##### Contributors- Dhananjay Saini, Dhruv Saxena.
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
