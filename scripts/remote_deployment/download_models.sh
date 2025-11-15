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

# MODEL CHECKPOINTS: PRE-TRAINED SNOMED ENCODERS

echo "Downloading SNOMED-tuned encoders ... "

./scripts/remote_deployment/download_gdrive_zip.sh "$SNOMED_URL" "$MODELS_DIR"

# MODEL CHECKPOINTS: PRE-TRAINED OnT ENCODERS

echo "Downloading pretrained OnT encoders ... "

./scripts/remote_deployment/download_gdrive_zip.sh "$ONT_URL" "$MODELS_DIR"

# MODEL CHECKPOINT: HiT Mixed Hard Negatives

echo "Downloading HiT-mixed-hard-negatives model checkpoint ... "

wget -P "$MODELS_DIR/snomed_models" https://ontozoo.io/models/HiT_mixed_hard.zip

echo "Unzipping HiT-mixed-hard model checkpoint ... "

unzip "$MODELS_DIR/snomed_models/HiT_mixed_hard.zip" -d "$MODELS_DIR/snomed_models"

echo "Removing HiT_mixed_hard.zip ... "

rm "$MODELS_DIR/snomed_models/HiT_mixed_hard.zip"

# MODEL CHECKPOINT: OnT-96

echo "Downloading OnT-96 model checkpoint ... "

wget -P "$MODELS_DIR/snomed_models" https://ontozoo.io/models/OnT-96-ckpt.zip

echo "Unzipping OnT-96 model checkpoint ... "

unzip "$MODELS_DIR/snomed_models/OnT-96-ckpt.zip" -d "$MODELS_DIR/snomed_models"

echo "Removing OnT-96-ckpt.zip ... "

rm "$MODELS_DIR/snomed_models/OnT-96-ckpt.zip"

echo "...Done! Models unpacked to $MODELS_DIR"
