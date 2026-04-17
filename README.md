# ml-infra

A production-grade ML inference infrastructure on **Google Kubernetes Engine (GKE)**, featuring an automated model optimization pipeline, NVIDIA Triton Inference Server, a REST backend, a web frontend, full CI/CD via Google Cloud Build, and observability through Prometheus + Grafana.

---

# 🚀 Highlights

**Latency Breakthrough:** Achieved **~50ms inference latency** (vs. expected ~500ms) through a combination of deep model optimizations and high‑performance serving architecture.

| Optimization | Impact |
|--------------|--------|
| ONNX conversion + quantization + INT8 calibration | Model size reduced from ~500 MB → ~100 MB |
| Dynamic input dimensions | Avoids unnecessary padding / recompilation |
| Single persistent gRPC connection | Eliminates per‑request connection setup overhead |
| `tritonclient.grpc.aio` (native async) | Bypasses `asyncio.to_thread` overhead; fully non‑blocking I/O |
| Health‑check exclusion from prediction path | Removes extraneous calls, cuts total inference time |

**Live Deployment & Observability**

- **Frontend Application:** [http://34.169.18.142/](http://34.169.18.142/)  
- **Grafana Dashboard:** [http://34.127.88.169/d/ad5mgqz/new-dashboard](http://34.127.88.169/d/ad5mgqz/new-dashboard?orgId=1&from=now-6h&to=now&timezone=browser)  
  Monitor Triton throughput, queue depth, GPU/CPU utilization, and FastAPI request latency in real time.

**Automated CI/CD (Google Cloud Build)**

1. **Model Pipeline Automation** – Converts, quantizes, validates, and uploads optimized ONNX models to GCS.
2. **Multi‑Stage Docker Builds** – Drastically reduces image size and build time via layer caching.
3. **Artifact Registry Push** – Images are stored securely in GCP Artifact Registry with unique `$BUILD_ID` tags.
4. **Full‑Stack Deployment** – Frontend, FastAPI backend, and Triton services are rolled out with Prometheus metrics scraping pre‑configured.

**Model Served:** `bvanaken/clinical-assertion-negation-bert` – demonstrates production‑ready NLP inference with sub‑100ms latency.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Components](#components)
  - [Backend](#backend)
  - [Frontend](#frontend)
  - [Model Pipeline](#model-pipeline)
  - [Kubernetes (k8s)](#kubernetes-k8s)
  - [CI/CD](#cicd)
  - [Dockerfiles](#dockerfiles)
- [Getting Started](#getting-started)
- [CI/CD Flow](#cicd-flow)
- [Observability](#observability)

---

## Architecture Overview

```
                        ┌─────────────────────────────────────────────┐
                        │              GKE Cluster                     │
                        │                                              │
  User ──► Frontend ──► │  Backend (FastAPI) ──► Triton Inference      │
                        │                         Server (gRPC)        │
                        │                                              │
                        │  Prometheus ◄── Triton Metrics               │
                        │  Grafana    ◄── Prometheus                   │
                        └─────────────────────────────────────────────┘
                                          ▲
                              Cloud Build CI/CD Pipeline
                              (Build → Optimize → Push → Deploy)
                                          ▲
                                    GCS Model Store
```

---

## Project Structure

```
ml-infra/
├── backend/                        # FastAPI inference backend
│   ├── main.py                     # API entry point & route definitions
│   ├── models.py                   # Request/response Pydantic models
│   ├── triton_client.py            # gRPC client to communicate with Triton
│   └── requirements.txt            # Python dependencies
│
├── ci-cd/
│   └── cloudbuild.yaml             # Google Cloud Build CI/CD pipeline
│
├── frontend/                       # Flask web frontend
│   ├── templates/
│   │   └── index.html              # Main UI page
│   ├── app.py                      # Flask application
│   └── requirements.txt            # Python dependencies
│
├── k8s/                            # Kubernetes manifests
│   ├── namespaces.yaml             # Namespace definitions for each service
│   ├── backend-deployment.yaml     # Backend Deployment & Service
│   ├── frontend-deployment.yaml    # Frontend Deployment & Service
│   ├── backendconfig.yaml          # GKE BackendConfig for frontend health checks
│   ├── triton-inference-straight.yaml  # Triton server Deployment & configuration
│   ├── triton-grpc-service.yaml    # Service to expose Triton on gRPC port
│   ├── triton-metrics-server.yaml  # Service to expose Triton metrics endpoint
│   ├── triton-service-monitor.yaml # Prometheus ServiceMonitor for Triton metrics
│   └── grafana-loadbalancer.yaml   # LoadBalancer Service to expose Grafana
│
├── model-pipeline/                 # Model optimization & upload pipeline
│   ├── run_pipeline.py             # Pipeline entry point
│   ├── convert_to_onnx.py          # Converts models to ONNX format
│   ├── optimize_model.py           # Quantization & INT8 calibration
│   ├── validate_model.py           # Validates optimized model performance
│   └── config.pbtxt                # Triton model serving configuration
│
├── Dockerfile.backend              # Docker image for the backend service
├── Dockerfile.frontend             # Docker image for the frontend service
├── Dockerfile.pipeline             # Docker image for the model pipeline
├── .dockerignore                   # Files excluded from Docker builds
├── .gcloudignore                   # Files excluded from Cloud Build uploads
└── .gitignore                      # Files excluded from Git commits
```

---

## Components

### Backend

**Location:** `ml-infra/backend/`

The backend is a **FastAPI** service that acts as the inference gateway between the frontend and the Triton Inference Server.

| File | Description |
|---|---|
| `main.py` | Defines API routes and application startup. Handles incoming inference requests and orchestrates calls to Triton. |
| `models.py` | Pydantic models for request validation and response serialization — ensures type-safe API contracts. |
| `triton_client.py` | gRPC client wrapper for communicating with the Triton Inference Server. Handles input tensor preparation, request dispatch, and response parsing. |
| `requirements.txt` | Python package dependencies for the backend service. |

---

### Frontend

**Location:** `ml-infra/frontend/`

A lightweight **Flask** web application that provides a browser-based interface to submit inference requests and view results.

| File | Description |
|---|---|
| `app.py` | Flask application that serves the UI and proxies requests to the backend API. |
| `templates/index.html` | Main HTML page — the user-facing inference interface. |
| `requirements.txt` | Python package dependencies for the frontend service. |

---

### Model Pipeline

**Location:** `ml-infra/model-pipeline/`

An end-to-end automated pipeline that takes a raw trained model, optimizes it for production inference, validates it, and uploads it to Google Cloud Storage (GCS) for Triton to serve.

| File | Description |
|---|---|
| `run_pipeline.py` | **Entry point.** Accepts model metadata as input, orchestrates the full pipeline — conversion → optimization → validation → GCS upload. |
| `convert_to_onnx.py` | Converts models from any framework (PyTorch, TensorFlow, etc.) to the **ONNX** format for a fast, framework-agnostic runtime. |
| `optimize_model.py` | Applies production optimizations: **model quantization** and **INT8 calibration** to reduce model size and improve inference latency. |
| `validate_model.py` | Benchmarks the optimized model against the original to measure accuracy/performance delta. Acts as a quality gate — only models within acceptable thresholds are promoted to production. |
| `config.pbtxt` | Triton Inference Server model configuration. Specifies how the model should be served — including model instances, batching strategy, input/output tensor definitions, and caching settings. |

---

### Kubernetes (k8s)

**Location:** `ml-infra/k8s/`

All Kubernetes manifests required to deploy and operate the full stack on GKE.

| File | Description |
|---|---|
| `namespaces.yaml` | Defines dedicated Kubernetes namespaces for logical isolation of each service (backend, frontend, triton, monitoring). |
| `backend-deployment.yaml` | Kubernetes `Deployment` and `Service` for the FastAPI backend. |
| `frontend-deployment.yaml` | Kubernetes `Deployment` and `Service` for the Flask frontend. |
| `backendconfig.yaml` | GKE `BackendConfig` resource that configures health check parameters for the frontend service on the GCP Load Balancer. |
| `triton-inference-straight.yaml` | Raw `Deployment` for the NVIDIA Triton Inference Server, including resource requests (GPU/CPU), volume mounts for the model store, and Triton startup arguments. |
| `triton-grpc-service.yaml` | Kubernetes `Service` that exposes the Triton server on its **gRPC port** (default: 8001) for the backend to consume. |
| `triton-metrics-server.yaml` | Kubernetes `Service` that exposes Triton's **Prometheus-compatible metrics endpoint** (default: 8002). |
| `triton-service-monitor.yaml` | Prometheus `ServiceMonitor` CRD that instructs Prometheus to scrape metrics from the Triton metrics service. |
| `grafana-loadbalancer.yaml` | Kubernetes `LoadBalancer` Service that exposes the Grafana dashboard externally for monitoring access. |

---

### CI/CD

**Location:** `ml-infra/ci-cd/`

| File | Description |
|---|---|
| `cloudbuild.yaml` | **Google Cloud Build** pipeline definition. On trigger, it: (1) builds Docker images for backend, frontend, and the Triton service; (2) runs the model optimization pipeline and uploads the optimized model to GCS; (3) pushes all images to **Google Container Registry (GCR)** with a unique build tag; (4) updates and applies Kubernetes deployment manifests to the **GKE cluster** with the latest image tags. |

---

### Dockerfiles

Located at the project root, each Dockerfile targets a specific service for containerization.

| File | Description |
|---|---|
| `Dockerfile.backend` | Builds the container image for the FastAPI backend service. |
| `Dockerfile.frontend` | Builds the container image for the Flask frontend service. |
| `Dockerfile.pipeline` | Builds the container image for the model optimization pipeline, used during the CI/CD build step. |
| `.dockerignore` | Specifies files and directories to exclude from Docker build contexts, keeping images lean. |
| `.gcloudignore` | Specifies files to exclude when uploading source to Cloud Build, reducing upload size and build time. |
| `.gitignore` | Specifies files to exclude from Git version control (credentials, build artifacts, etc.). |

---

## Getting Started

### Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- A GKE cluster with GPU node pools (for Triton)
- A GCS bucket for model storage
- Prometheus + Grafana installed on the cluster (e.g., via `kube-prometheus-stack`)

### Deploy to GKE

```bash
# 1. Create namespaces
kubectl apply -f k8s/namespaces.yaml

# 2. Deploy Triton Inference Server
kubectl apply -f k8s/triton-inference-straight.yaml
kubectl apply -f k8s/triton-grpc-service.yaml
kubectl apply -f k8s/triton-metrics-server.yaml

# 3. Deploy Backend and Frontend
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/backendconfig.yaml

# 4. Deploy Monitoring
kubectl apply -f k8s/triton-service-monitor.yaml
kubectl apply -f k8s/grafana-loadbalancer.yaml
```

### Run the Model Pipeline Locally

```bash
cd model-pipeline
pip install -r ../backend/requirements.txt
python run_pipeline.py --model-name <your_model> --model-path <path/to/model>
```

---

## CI/CD Flow

Every push to the main branch triggers `ci-cd/cloudbuild.yaml`, which executes the following steps in order:

```
1. Build Docker images  ──►  backend, frontend, pipeline
2. Run model-pipeline   ──►  convert → optimize → validate → upload to GCS
3. Push images to GCR   ──►  tagged with unique $BUILD_ID
4. Deploy to GKE        ──►  update k8s manifests with new image tags & apply
```

---

## Observability

| Component | Purpose |
|---|---|
| Triton Metrics (`triton-metrics-server.yaml`) | Exposes model throughput, latency, queue depth, and GPU utilization |
| Prometheus (`triton-service-monitor.yaml`) | Scrapes and stores Triton metrics |
| Grafana (`grafana-loadbalancer.yaml`) | Visualizes metrics via dashboards; exposed externally via LoadBalancer |