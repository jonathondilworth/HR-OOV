SHELL := /usr/bin/env bash
.DEFAULT_GOAL := all

IMAGE ?= hroov:gpu

.PHONY: init env models embeddings single-target multi-target tests all

init:
	@echo "[INIT] Initialising project enviornment variables and .env"
	./scripts/remote_deployment/init.sh

env:
	@echo "[ENV] Bootstrapping environment dependencies..."
	./scripts/remote_deployment/bootstrap.sh

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

all: init env models embeddings single-target multi-target
	@echo "[ALL] Finished running entire pipeline!"

default: all
