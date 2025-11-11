#!/usr/bin/env bash
set -euo pipefail

# Load .env if present (provides SNOMED_MODELS_URL, ONT_PRETRAINED_MODEL_URL)
if [[ -f .env ]]; then
  set -a
  . .env
  set +a
fi

MODELS_DIR="./models"
SNOMED_URL="${SNOMED_MODELS_URL:-https://drive.google.com/uc?id=1cQOqFVOHqBKkSirepzF7ga6mRYPP-LnT}"
ONT_URL="${ONT_PRETRAINED_MODEL_URL:-https://drive.google.com/uc?id=1t9xWcLHoEE55F0bOPMCw5jltWBxHc2vR}"

mkdir -p "$MODELS_DIR"

echo "Downloading SNOMED-tuned encoders ... "

./scripts/remote_deployment/download_gdrive_zip.sh "$SNOMED_URL" "$MODELS_DIR"

echo "Downloading pretrained OnT encoders ... "

./scripts/remote_deployment/download_gdrive_zip.sh "$ONT_URL" "$MODELS_DIR"

echo "Downloading OnT-LG model checkpoint ... "

wget -P "$MODELS_DIR/snomed_models" https://ontozoo.io/models/OnT-LG.zip

echo "Unzipping OnT-LG model checkpoint ... "

unzip "$MODELS_DIR/snomed_models/OnT-LG.zip" -d "$MODELS_DIR/snomed_models"

echo "Removing OnT-LG.zip ... "

rm "$MODELS_DIR/snomed_models/OnT-LG.zip"

echo "...Done! Models unpacked to $MODELS_DIR"
