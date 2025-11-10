#!/usr/bin/env bash
set -Eeuo pipefail

echo "[INFO] Initialising project!"

echo "[WARNING] DO NOT call from anywhere besides the root dir; or else everything will break! You have been warned!"

# WARNING: do not call from anywhere besides the root dir
# (...else everything will break! You have been warned!)
project_root="$PWD"

echo "[INFO] Exposing lib to pythonpath"

# exposes local forks in lib (neccesary for local training support)
# NOTE: removing local forks and relying on packaged versions will break training support
export PYTHONPATH="$project_root/lib${PYTHONPATH:+:$PYTHONPATH}"

echo "[INFO] Checking .env exists (if not, writing it)"

# if .env doesn't already exist, create it
touch .env

echo "[INFO] Saving sampling procedural data inside .env (if not already set)"

# for later scripts, we require setting the SAMPLING_PROCEDURE
# default to deterministic, but if set to random, do not overwrite
if ! grep -q '^SAMPLING_PROCEDURE=' .env 2>/dev/null; then
  echo "SAMPLING_PROCEDURE=deterministic" >> .env
fi

# set the number of instances to sample (default=50)
if ! grep -q '^SAMPLE_N=' .env 2>/dev/null; then
  echo "SAMPLE_N=50" >> .env
fi

echo "[INFO] Saving project root inside .env (if not already set)"

# orchestration process requires the root dir to be stored as an ENV variable
if ! grep -q '^project_root=' .env 2>/dev/null; then
  echo "Setting project_root (.env) to: $project_root"
  echo "project_root=$project_root" >> .env
fi

# wandb.ai (weights and biases); set WANDB_MODE=['online', 'offline', 'disabled', 'shared'] (disable by default)
if ! grep -q '^WANDB_MODE=' .env 2>/dev/null; then
  echo "Setting WANDB_MODE (.env) to: disabled"
  echo "WANDB_MODE=disabled" >> .env
fi

# encoders for end-to-end reproducability (ablation)
if ! grep -q '^ONT_PRETRAINED_MODEL_URL=' .env 2>/dev/null; then
  echo "Setting ONT_PRETRAINED_MODEL_URL (.env) to: https://drive.google.com/uc?id=1t9xWcLHoEE55F0bOPMCw5jltWBxHc2vR"
  echo "ONT_PRETRAINED_MODEL_URL=https://drive.google.com/uc?id=1t9xWcLHoEE55F0bOPMCw5jltWBxHc2vR" >> .env
fi

# encoders for end-to-end reproducability (SNOMED domain-tuned)
if ! grep -q '^SNOMED_MODELS_URL=' .env 2>/dev/null; then
  echo "Setting SNOMED_MODELS_URL (.env) to: https://drive.google.com/uc?id=1cQOqFVOHqBKkSirepzF7ga6mRYPP-LnT"
  echo "SNOMED_MODELS_URL=https://drive.google.com/uc?id=1cQOqFVOHqBKkSirepzF7ga6mRYPP-LnT" >> .env
fi

# since most people running this build script are unlikely to have a copy of SNOMED / a SNOMED license
# a failover (older release) is available @ https://zenodo.org/doi/10.5281/zenodo.10511042
# downloadable (as a .zip) @ https://zenodo.org/records/14036213/files/ontologies.zip?download=1

if ! grep -q '^NHS_API_KEY=' .env 2>/dev/null; then
  echo "Setting ALT_SNOMED_ONTOLOGY_URL (.env) to: https://zenodo.org/records/14036213/files/ontologies.zip?download=1"
  echo "ALT_SNOMED_ONTOLOGY_URL=https://zenodo.org/records/14036213/files/ontologies.zip?download=1" >> .env
fi

echo "[INFO] Initialisation finished."
