# HR-OOV

SNOMED CT is a biomedical ontology with a hierarchical representation of large-scale concepts. Knowledge retrieval in SNOMED CT is critial for its application, but often proves challanging due to language ambiguity, synonyms, polysemies and so on. This problem is exacerbated when the queries are out-of-vocabulary (OOV), i.e., having no equivalent matchings in the ontology. In this work, we focus on the problem of hierarchical concept retrieval from SNOMED CT with OOV queries, and propose an approach based on language model-based ontology embeddings. For evaluation, we construct OOV queries annotated against SNOMED CT concepts, testing the retrieval of the most direct subsumers and their less relevant ancestors. We find that our method outperforms the baselines including SBERT and two lexical matching methods. While evaluated against SNOMED CT, the approach is generalisable and can be extended to other ontologies. We release code, tools, and evaluation datasets at [https://github.com/jonathondilworth](HR-OOV).

## Features

* Create new datasets to test agaisnt using `make sample` *(requires manual annotation)*.

* Re-use the [provided retrievers](./src/hroov/utils/retrievers.py) and [gpu_retrievers](./src/hroov/utils/gpu_retrievers.py) for knowledge retrieval. These support:
		
	* TF-IDF, BM25
		
	* SBERT
		
	* HiT and OnT *(with hyperbolic distance and/or entity & concept subsumption)*

* Review experimental results.

	* Published within the [logs](./logs) folder, and can be viewed within the [included notebook](./notebooks/scratch_notebook.ipynb).

## Usage

See the included [Makefile](./Makefile) and the note under [reproducability](#Reproducibility) on deployment.

* To initialise the repo on a remote machine, clone the repository with

	* `git clone https://github.com/jonathondilworth/HR-OOV.git`,

	* and `cd HR-OOV`. Then run `make init` and `make env`.

* To process SNOMED CT, run `make download-snomed` and `make process-snomed`.

* To process MIRAGE, run `make download-mirage` and `make process-mirage`.

* To create new datasets, set `SAMPLING_PROCEDURE=random` in `.env` and run `make sample`.

See details under [reproducability](#Reproducibility) for re-creating experimental results.

*(Instructions on local execution with docker coming soon...)*

## Reproducibility

**For end-to-end reproducability, we strongly suggest running the included `Makefile` within a fresh VM instance and is presently used for deploying to remote cloud VMs *(support for docker will be included shortly)*.**

To reproduce experimental results, add your `NHS_API_KEY` to `.env` and set `SAMPLING_PROCEDURE=deterministic`, as shown in [the example env file](./env.example).

Then run `make`. 

This procedure will:

1. Initialises the project using [init.sh](./scripts/remote_deployment/init.sh).
2. Configures the environment with [env.sh](./scripts/remote_deployment/env.sh).
3. Downloads and processes the September 2025 release of SNOMED CT.
		
	* This will failover to a publicly available version if no `NHS_API_KEY` has been provided.
		
	* Failing to provide the `NHS_API_KEY` will result in small variation in the results.

4. Downloads embedding models.
5. Produces embeddings for experiments.
6. Runs single and multiple target experiments.

See the [Makefile](./Makefile) and [scripts folder](./scripts) for the specific implementation.

### Environment

- OS: Ubuntu 22.04
- Python: 3.12
- NVCC: 12.9

### Hardware

- NVIDIA GPU: H200
- vCPUs: 24
- Memory: 240 GB

## License

All source code is licensed under the MIT License (see [LICENSE](./LICENSE)).

### SNOMED CT

This repository does **not** redistribute the full SNOMED CT release or any substantial portion of it. Scripts in this repository may download SNOMED CT content from **existing, publicly available** datasets. The small evaluation subset packaged within this repository is derived from SNOMED CT International Edition (release 2025/10/01).

This material includes SNOMED Clinical Terms® (SNOMED CT®) which is used by permission of the International Health Terminology Standards Development Organisation (IHTSDO). All rights reserved. SNOMED CT®, was originally created by The College of American Pathologists. "SNOMED" and "SNOMED CT" are registered trademarks of the IHTSDO.
