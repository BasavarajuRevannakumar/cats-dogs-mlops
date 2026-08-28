#!/bin/bash

set -e

BASE_URL="${1:-http://192.168.49.2:32205}"

echo "======================================"
echo "Cats-Dogs API Smoke Test"
echo "Base URL: $BASE_URL"
echo "======================================"

echo
echo "1. Health check"

HEALTH_RESPONSE=$(curl -fsS "$BASE_URL/health")

echo "Response:"
echo "$HEALTH_RESPONSE"

echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'
echo "$HEALTH_RESPONSE" | grep -q '"model_loaded":true'

echo "Health check: PASS"

echo
echo "2. Prediction check"

# Create a valid 1x1 PNG image.
# This keeps the smoke test self-contained and does not depend on the dataset.
IMAGE_FILE=$(mktemp --suffix=.png)

trap 'rm -f "$IMAGE_FILE"' EXIT

echo 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=' \
  | base64 -d > "$IMAGE_FILE"

PREDICTION_RESPONSE=$(curl -fsS \
  -X POST \
  "$BASE_URL/predict" \
  -F "file=@$IMAGE_FILE;type=image/png")

echo "Response:"
echo "$PREDICTION_RESPONSE"

echo "$PREDICTION_RESPONSE" | grep -q '"prediction"'
echo "$PREDICTION_RESPONSE" | grep -q '"label"'
echo "$PREDICTION_RESPONSE" | grep -q '"probability"'

echo "Prediction check: PASS"

echo
echo "======================================"
echo "ALL SMOKE TESTS PASSED"
echo "======================================"
