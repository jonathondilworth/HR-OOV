SHELL := /usr/bin/env bash
.DEFAULT_GOAL := all

IMAGE ?= hroov:gpu

.PHONY: init env download-snomed process-snomed download-mirage process-mirage sample hit-data hit-data-custom ont-data hit-train ont-train models embeddings single-target multi-target tests all

init:
	@echo "[INIT] Initialising project enviornment variables and .env"
	./scripts/remote_deployment/init.sh

env:
	@echo "[ENV] Bootstrapping environment dependencies..."
	./scripts/remote_deployment/bootstrap.sh

download-snomed:
	@echo "[SNOMED] Downloading SNOMED CT (ensure NHS_API_KEY is set in .env)"
	./scripts/remote_deployment/download_snomed.sh

process-snomed:
	@echo "[PROCESS-SNOMED] Processing SNOMED CT..."
	./scripts/remote_deployment/process_snomed.sh

download-mirage:
	@echo "[MIRAGE] Downloading MIRAGE ... "
	./scripts/remote_deployment/download_mirage.sh

process-mirage:
	@echo "[MIRAGE] Processing MIRAGE ... (Question Extract & NER)"
	./scripts/remote_deployment/process_mirage.sh

sample:
	@echo "[SAMPLE] Sampling processed datasets (SNOMED CT and MIRAGE)..."
	./scripts/remote_deployment/process_diff_and_sample.sh

hit-data:
	@echo "[HIT] Building HiT dataset (running pipeline)..."
	./scripts/remote_deployment/build_hit_data.sh

hit-data-custom:
	@echo "[HIT] Building HiT dataset using custom ontology (running pipeline)..."
	./scripts/remote_deployment/build_hit_data_custom.sh

ont-data:
	@echo "[ONT] Building OnT dataset (running pipeline)..."
	./scripts/remote_deployment/build_ont_data.sh

ont-data-custom:
	@echo "[ONT] Building OnT dataset for CUSTOM data (running pipeline)..."
	./scripts/remote_deployment/build_ont_data_custom.sh

ont-data-custom-hard:
	@echo "[ONT] Building OnT dataset for CUSTOM data USING HARD NEGATIVES FOR NF1 (running pipeline)..."
	./scripts/remote_deployment/build_ont_data_custom_hard.sh

hit-train:
	@echo "[HIT] Starting HiT training..."
	./scripts/remote_deployment/train_hit.sh

hit-train-custom:
	@echo "[HIT] Starting HiT training with custom dataset..."
	./scripts/remote_deployment/train_hit_custom.sh

ont-train:
	@echo "[ONT] Starting OnT training..."
	./scripts/remote_deployment/train_ont.sh

ont-train-custom:
	@echo "[ONT] Starting OnT training of custom model..."
	./scripts/remote_deployment/train_ont_custom.sh

ont-eval-tmp:
	@echo "[ONT] EVAL TMP..."
	./scripts/remote_deployment/eval_ont_tmp.sh

models:
	@echo "[MODELS] Fetching both SNOMED-tuned and pretrained encoders ..."
	./scripts/remote_deployment/download_models.sh

embeddings:
	@echo "[EMBEDDINGS] Generating embeddings ... "
	./scripts/remote_deployment/produce_embeddings.sh

single-target:
	@echo "[SINGLE-TARGET] Running single target experiments ... "
	./scripts/remote_deployment/single_target.sh

multi-target:
	@echo "[MULTI-TARGET] Running multiple target experiments ... "
	./scripts/remote_deployment/multitarget.sh

all: init env download-snomed process-snomed models embeddings single-target multi-target
	@echo "[ALL] Finished running entire pipeline!"

default: all
