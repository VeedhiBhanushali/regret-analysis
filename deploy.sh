#!/bin/bash
set -e

PROJECT_ID=${1:?"Usage: ./deploy.sh <gcp-project-id>"}
REGION="us-central1"
SERVICE_NAME="regret-inference"

echo "=== Step 1: Train and export models ==="
python inference/train_and_export.py

echo ""
echo "=== Step 2: Deploy App Engine (website) ==="
gcloud app deploy app.yaml --project="$PROJECT_ID" --quiet

echo ""
echo "=== Step 3: Upload data to Firestore ==="
python src/upload_to_firestore.py

echo ""
echo "=== Step 4: Build and deploy Cloud Run (inference) ==="
cd inference
gcloud builds submit --tag "gcr.io/$PROJECT_ID/$SERVICE_NAME" --project="$PROJECT_ID"
gcloud run deploy "$SERVICE_NAME" \
  --image "gcr.io/$PROJECT_ID/$SERVICE_NAME" \
  --platform managed \
  --region "$REGION" \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --allow-unauthenticated \
  --project="$PROJECT_ID"
cd ..

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --platform managed --region "$REGION" --project="$PROJECT_ID" \
  --format="value(status.url)")

echo ""
echo "=== Deployment complete ==="
echo "App Engine: https://$PROJECT_ID.uc.r.appspot.com"
echo "Cloud Run:  $SERVICE_URL"
echo ""
echo "IMPORTANT: Update the ANALYZER_API variable in index.html with:"
echo "  $SERVICE_URL"
