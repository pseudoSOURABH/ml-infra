#!/bin/bash
# Manual deployment script for testing/rollback
set -euo pipefail

REGION="${REGION:-us-central1}"
CLUSTER="${CLUSTER:-ml-platform-cluster}"
PROJECT="${PROJECT:-}"
NAMESPACE="${NAMESPACE:-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

if [ -z "$PROJECT" ]; then
    echo "ERROR: PROJECT environment variable required"
    exit 1
fi

echo "🚀 Deploying to $CLUSTER in $REGION..."

# Get credentials
gcloud container clusters get-credentials "$CLUSTER" --region "$REGION" --project "$PROJECT"

# Export variables for envsubst
export PROJECT_ID="$PROJECT"
export REGION
export IMAGE_TAG

# Apply manifests in order
echo "📦 Applying namespaces..."
kubectl apply -f k8s/namespaces.yaml

echo "⚙️  Applying KServe config..."
kubectl apply -f k8s/kserve-config.yaml

echo "🤖 Deploying Triton inference service..."
envsubst < k8s/triton-inference.yaml | kubectl apply -f -

echo "🔌 Deploying backend..."
envsubst < k8s/backend-deployment.yaml | kubectl apply -f -

echo "🎨 Deploying frontend..."
envsubst < k8s/frontend-deployment.yaml | kubectl apply -f -

echo "🌐 Applying ingress..."
envsubst < k8s/ingress.yaml | kubectl apply -f -

# Wait for rollouts
echo "⏳ Waiting for deployments..."
kubectl rollout status deployment/backend -n api --timeout=300s
kubectl rollout status deployment/frontend -n frontend --timeout=300s

# Check KServe service
echo "🔍 Checking InferenceService status..."
kubectl get inferenceservice clinical-assertion -n triton -o yaml | grep -A5 "status:" || true

echo "✅ Deployment complete!"
echo "📊 Check status:"
echo "  kubectl get pods -A -l app in (backend,frontend)"
echo "  kubectl get inferenceservice -n triton"
echo "🌐 Access: https://${DOMAIN_NAME:-<your-domain>}"