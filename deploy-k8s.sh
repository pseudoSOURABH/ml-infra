#!/usr/bin/env bash
# deploy-k8s.sh
# Usage: deploy-k8s.sh <CLUSTER> <REGION> <PROJECT_ID> <IMAGE_TAG> <MODEL_GCS_URI>
#
# IMAGE_TAG    — rich tag built by cloudbuild compute-image-tag step,
#                e.g.  a1b2c3d4-feat-deploy-model-20260413
# MODEL_GCS_URI — canonical GCS path written by check-model-cache step and
#                 uploaded to by run_pipeline.py, e.g.
#                 gs://my-project-ml-models/clinical_assertion
#
# Both values arrive from /workspace/*.env files sourced in the Cloud Build
# deploy step, so they are always in sync with what was actually built/uploaded
# in the same pipeline run.

set -euo pipefail

CLUSTER="${1:?CLUSTER is required}"
REGION="${2:?REGION is required}"
PROJECT_ID="${3:?PROJECT_ID is required}"
IMAGE_TAG="${4:?IMAGE_TAG is required}"
MODEL_GCS_URI="${5:?MODEL_GCS_URI is required}"

MANIFESTS_DIR="${MANIFESTS_DIR:-$(dirname "$0")/../k8s}"

echo "=== deploy-k8s.sh ==="
echo "  Cluster:       ${CLUSTER}"
echo "  Region:        ${REGION}"
echo "  Project:       ${PROJECT_ID}"
echo "  Image tag:     ${IMAGE_TAG}"
echo "  Model GCS URI: ${MODEL_GCS_URI}"
echo "  Manifests dir: ${MANIFESTS_DIR}"
echo ""

# ── Authenticate to GKE ──────────────────────────────────────────────────────
gcloud container clusters get-credentials "${CLUSTER}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}"

# ── Helper: replace placeholders and apply a manifest ───────────────────────
apply_manifest() {
  local template="$1"
  echo "→ Applying ${template}"
  sed \
    -e "s|@@REGION@@|${REGION}|g" \
    -e "s|@@PROJECT_ID@@|${PROJECT_ID}|g" \
    -e "s|@@IMAGE_TAG@@|${IMAGE_TAG}|g" \
    -e "s|@@MODEL_GCS_URI@@|${MODEL_GCS_URI}|g" \
    "${template}" \
  | kubectl apply -f -
}

# ── Apply all manifests ──────────────────────────────────────────────────────
apply_manifest "${MANIFESTS_DIR}/backend.yaml"
apply_manifest "${MANIFESTS_DIR}/frontend.yaml"
apply_manifest "${MANIFESTS_DIR}/inference-service.yaml"

# ── Wait for rollouts ────────────────────────────────────────────────────────
echo ""
echo "⏳ Waiting for rollouts..."
kubectl rollout status deployment/clinical-backend  -n api      --timeout=300s
kubectl rollout status deployment/frontend          -n frontend  --timeout=300s

echo ""
echo "✓ All rollouts complete."
echo "  Deployed image tag : ${IMAGE_TAG}"
echo "  Triton storageUri  : ${MODEL_GCS_URI}"