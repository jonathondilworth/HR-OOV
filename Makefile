SHELL := /usr/bin/env bash
.DEFAULT_GOAL := all

IMAGE ?= hroov:gpu

.PHONY: init env models embeddings all

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

all: init env models embeddings
	@echo "[ALL] Finished running entire pipeline!"

default: all
