Here is the complete, combined `README.md` with the original content preserved in full and the new **GCP Resource Configuration** section appended at the bottom.

```markdown
# ml-infra

A production-grade ML inference infrastructure on **Google Kubernetes Engine (GKE)**, featuring an automated model optimization pipeline, NVIDIA Triton Inference Server, a REST backend, a web frontend, full CI/CD via Google Cloud Build, and observability through Prometheus + Grafana.

---

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
- [GCP Resource Configuration](#gcp-resource-configuration)

---

## Architecture Overview

### Traffic Flow

```
                                    ┌──────────────────────────────────────────────────────────────────┐
                                    │                        GKE Cluster                                │
                                    │                                                                   │
                                    │  ┌─────────────────────┐       ┌──────────────────────────────┐  │
                                    │  │   frontend namespace │       │       api namespace           │  │
  User                              │  │                     │       │                              │  │
   │                                │  │  ┌───────────────┐  │ HTTP  │  ┌────────────────────────┐  │  │
   │  HTTPS to LoadBalancer IP      │  │  │   Frontend    │──┼───────┼─►│       Backend          │  │  │
   └───────────────────────────────►│  │  │  (Flask)      │  │       │  │      (FastAPI)         │  │  │
                                    │  │  └───────────────┘  │       │  └──────────┬─────────────┘  │  │
                                    │  └─────────────────────┘       │             │ gRPC :8001     │  │
                                    │                                 └─────────────┼────────────────┘  │
                                    │                                               │                   │
                                    │                                 ┌─────────────▼────────────────┐  │
                                    │                                 │      triton namespace         │  │
                                    │                                 │                              │  │
                                    │                                 │  ┌────────────────────────┐  │  │
                                    │                                 │  │    Triton Inference    │  │  │
                                    │                                 │  │       Server           │  │  │
                                    │                                 │  │  gRPC :8001            │  │  │
                                    │                                 │  │  metrics :8002         │──┼──┼──┐
                                    │                                 │  └────────────────────────┘  │  │  │
                                    │                                 └──────────────────────────────┘  │  │
                                    │                                                                   │  │
                                    │  ┌────────────────────────────────────────────────────────────┐  │  │
                                    │  │                  monitoring namespace                        │  │  │
                                    │  │                                                             │  │  │
                                    │  │  Triton Metrics Svc (:8002) ◄──────────────────────────────┼──┼──┘
                                    │  │          │                                                  │  │
                                    │  │          │ scrape (ServiceMonitor)                          │  │
                                    │  │          ▼                                                  │  │
                                    │  │      Prometheus ──────────► Grafana (LoadBalancer) ◄────── User
                                    │  │                    query                                    │  │
                                    │  └────────────────────────────────────────────────────────────┘  │
                                    └──────────────────────────────────────────────────────────────────┘
```

### CI/CD Pipeline (Cloud Build)

Every push to the main branch triggers `ci-cd/cloudbuild.yaml`, which runs the following steps in sequence:

```
  Step 1                   Step 2                        Step 3
  ┌──────────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
  │ Build pipeline   │     │ Run model pipeline     │     │   Upload to GCS      │
  │ Docker image     │────►│                       │────►│                      │
  │ (Dockerfile      │     │  raw model            │     │  optimized ONNX      │
  │  .pipeline)      │     │     │ convert_to_onnx  │     │  model + config.pbtxt│
  └──────────────────┘     │     ▼                 │     └──────────┬───────────┘
                           │  ONNX format          │                │ model load
                           │     │ optimize_model   │                ▼
                           │     ▼                 │        ┌───────────────┐
                           │  quantized + INT8     │        │  GCS bucket   │
                           │     │ validate_model  │        │  (model store)│
                           │     ▼                 │        └───────┬───────┘
                           │  performance check    │                │
                           └───────────────────────┘                │ Triton reads
                                                                     │ model on startup
  Step 4                   Step 5                                    ▼
  ┌──────────────────────────────┐     ┌──────────────────────────────────────┐
  │ Build & push service images  │     │    Deploy to GKE                      │
  │                              │     │                                       │
  │  backend  → GCR              │────►│  kubectl apply k8s/ manifests         │
  │  frontend → GCR              │     │  with new image tag ($BUILD_ID)       │
  │  triton   → GCR              │     │  rolling update on each service       │
  └──────────────────────────────┘     └──────────────────────────────────────┘
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

---

## GCP Resource Configuration

This section documents the Google Cloud Platform (GCP) infrastructure configuration, including IAM, networking, compute resources, and additional platform services like KServe and Prometheus.

### Identity and Access Management (IAM)

#### 1. Application Service Account (Workload Identity)

**GSA:** `ml-platform-sa@${PROJECT_ID}.iam.gserviceaccount.com`

**Purpose:** Enables GKE workloads to securely access Google Cloud APIs via Workload Identity, eliminating the need for static JSON keys.

| Role | IAM Permission | Use Case |
|------|----------------|----------|
| Artifact Registry Reader | `roles/artifactregistry.reader` | Image pulls during Pod initialization. |
| Logs Writer | `roles/logging.logWriter` | Application telemetry and structured logging. |
| Monitoring Metric Writer | `roles/monitoring.metricWriter` | Custom metrics for observability. |
| Storage Object Admin | `roles/storage.objectAdmin` | CRUD operations on ML model artifacts and data. |

#### 2. CI/CD Pipeline Service Account

**Purpose:** Automates the build, test, and deployment lifecycle (e.g., Cloud Build).

| Role | IAM Permission | Use Case |
|------|----------------|----------|
| Artifact Registry Writer | `roles/artifactregistry.writer` | Pushing Docker images post-build. |
| GKE Developer | `roles/container.developer` | Orchestrating cluster workloads and deployments. |
| Secret Manager Accessor | `roles/secretmanager.secretAccessor` | Retrieving build-time credentials and secrets. |
| Storage Admin | `roles/storage.admin` | Managing CI/CD assets, logs, and state files. |
| Logs Writer | `roles/logging.logWriter` | Auditability of pipeline execution stages. |

#### 3. Compute Engine Default Service Account

**Identity:** `{PROJECT_NUMBER}-compute@developer.gserviceaccount.com`

**Context:** Serves as the default Node Identity for the GKE cluster.

- **Node Lifecycle:** Used by the GKE node agent for Compute Engine API interactions.
- **System Telemetry:** Facilitates reporting of VM-level health and metrics.
- **Legacy Scopes:** Utilized for default resource access during initial cluster provisioning.

> **Security Note:** It is recommended to replace this with a Custom Node Service Account in production to enforce the Principle of Least Privilege.

### IAM Configuration and Workload Identity Implementation

#### 1. Application Service Account Authorization

Grant required resource access to the Google Service Account (GSA).

```bash
# Roles: Artifact Registry Reader, Logs Writer, Monitoring Metric Writer, Storage Object Admin
for role in artifactregistry.reader logging.logWriter monitoring.metricWriter storage.objectAdmin; do
  gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:ml-platform-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/$role"
done
```

#### 2. CI/CD Pipeline Service Account Authorization

Grant deployment and registry permissions to the CI/CD service account.

```bash
# Roles: Artifact Registry Writer, Container Developer, Secret Manager Accessor, Storage Admin
for role in artifactregistry.writer container.developer secretmanager.secretAccessor storage.admin; do
  gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:cicd-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/$role"
done
```

#### 3. Workload Identity & KSA Integration

Establish the security binding between the Kubernetes Service Account (KSA) and the GSA.

```bash
# Bind GSA to KSA
gcloud iam service-accounts add-iam-policy-binding ml-platform-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[${K8S_NAMESPACE}/${KSA_NAME}]"

# Create and Annotate KSA
kubectl create serviceaccount ${KSA_NAME} --namespace ${K8S_NAMESPACE}

kubectl annotate serviceaccount ${KSA_NAME} \
    --namespace ${K8S_NAMESPACE} \
    iam.gke.io/gcp-service-account=ml-platform-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

#### 4. Verification

Validate policy application for the specified identity.

```bash
gcloud projects get-iam-policy ${PROJECT_ID} \
    --flatten="bindings[].members" \
    --format="table(bindings.role)" \
    --filter="bindings.members:ml-platform-sa@${PROJECT_ID}.iam.gserviceaccount.com"
```

### Infrastructure Resource Provisioning

The following commands utilize the Google Cloud CLI to provision the container registry and object storage resources.

#### 1. Artifact Registry

Establish a secure, regional repository for Docker container images.

```bash
gcloud artifacts repositories create ml-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Production Docker repository for ML platform"
```

#### 2. Cloud Storage (GCS)

Create a regional storage bucket for ML artifacts and data. The `--uniform-bucket-level-access` flag is included to align with security best practices for IAM-based access control.

```bash
# Replace [BUCKET_NAME] with a globally unique identifier
export BUCKET_NAME="ml-artifacts-${PROJECT_ID}"

gcloud storage buckets create gs://${BUCKET_NAME} \
    --location=us-central1 \
    --uniform-bucket-level-access
```

### GKE Cluster and Compute Configuration

This section outlines the GKE cluster and node pool configuration. Due to quota limitations within the GCP Free Tier regarding GPU availability, the infrastructure is optimized for CPU-based compute.

#### 1. Control Plane & System Node Pool

The cluster is initialized with a managed system node pool to handle core Kubernetes services and cluster administration.

```bash
gcloud container clusters create ${CLUSTER_NAME} \
    --zone=us-central1-a \
    --workload-pool=${PROJECT_ID}.svc.id.goog \
    --enable-ip-alias \
    --release-channel=regular \
    --machine-type=e2-standard-2 \
    --num-nodes=1 \
    --disk-type=pd-standard \
    --disk-size=30 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=1
```

#### 2. Application Node Pool (CPU Optimized)

A dedicated node pool is provisioned for application workloads. In the absence of GPU-accelerated nodes, high-concurrency CPU nodes are utilized to support ML inference and platform services.

```bash
gcloud container node-pools create cpu-app-pool \
    --cluster=${CLUSTER_NAME} \
    --zone=us-central1-a \
    --machine-type=e2-standard-2 \
    --num-nodes=2 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=4 \
    --disk-type=pd-standard \
    --disk-size=30
```

#### 3. Static External IP for Frontend

Create a static IP address to expose the frontend service to the internet.

```bash
gcloud compute addresses create frontend-static-ip \
    --addresses 34.169.18.142 \
    --region us-west1
```

### Monitoring Stack Installation (Prometheus + Grafana)

```bash
# Add the Prometheus Community Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Create a dedicated namespace
kubectl create namespace monitoring

# Install the stack with persistent storage
helm install prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set grafana.adminPassword=your-admin-password
```

### Instrument FastAPI Backend with Prometheus Metrics

Your FastAPI backend needs instrumentation to expose HTTP request metrics (request count, latency histograms, etc.). Use the `prometheus-fastapi-instrumentator` library for minimal configuration.

**Add to `backend/requirements.txt`:**
```
prometheus-fastapi-instrumentator==2.0.0
```

**Update `backend/main.py` to add instrumentation:**

Add these lines near the top (after `app = FastAPI(...)`):

```python
from prometheus_fastapi_instrumentator import Instrumentator

# ... existing FastAPI app creation ...

# Instrument the app with default metrics (request counts, latency histograms)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
```

### KServe Deployment and Configuration

This documentation covers the deployment of KServe as the core inference orchestration layer. It is critical to select the appropriate deployment mode based on your service-level agreements (SLAs) for latency.

The KServe `quick_install.sh` script facilitates the automated deployment of the inference platform. For production environments where granular control and minimal latency are required, **RawDeployment** mode is the recommended configuration.

#### Deployment Command

```bash
# Execute quick install with RawDeployment flag for high-performance requirements
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.10/hack/quick_install.sh" | bash -s -- -r
```

#### Components and Infrastructure

The automated installation provisions the following stack:

- **Istio:** Manages the service mesh for secure, observable traffic routing.
- **Cert-manager:** Automates TLS certificate issuance and rotation.
- **KServe Controllers:** Deploys the Custom Resource Definitions (CRDs) and operators required to manage InferenceServices.

#### Deployment Mode Comparison

| Flag | Mode | Operational Impact |
|------|------|-------------------|
| `-s` | Serverless | Utilizes Knative for request-based autoscaling (Scale-to-Zero). |
| `-r` | RawDeployment | Utilizes native Kubernetes Deployments. Bypasses Knative and queue-proxy sidecars. |
| `-u` | Uninstall | Systematic removal of all KServe-related resources and dependencies. |

> **⚠️ CRITICAL OPERATIONAL ADVISORY**  
> **Latency Constraints in Serverless Mode:**  
> While Knative (Serverless mode) provides significant advantages for cost management and automated scaling, it is **not recommended** for real-time, low-latency inference workloads.  
> Knative introduces a `queue-proxy` sidecar that intercepts all incoming traffic. If a pod is in a "cold" state or scaling up, requests are held in a queue until the container is ready, leading to significant latency spikes.  
> **Production Recommendation:**  
> To achieve deterministic, low-latency response times, utilize **RawDeployment** mode. This ensures traffic reaches the inference container directly without sidecar-induced queuing. For production systems requiring high availability, avoid "Scale-to-Zero" and maintain a minimum "warm" pod count or implement custom pre-warming strategies.
```

This full `README.md` now contains everything you provided—the original project documentation plus the new GCP configuration details in a dedicated section at the end. You can copy and paste it directly into your repository.