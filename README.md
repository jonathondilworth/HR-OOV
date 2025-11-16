# HR-OOV

SNOMED CT is a biomedical ontology with a hierarchical representation of large-scale concepts. Knowledge retrieval in SNOMED CT is critial for its application, but often proves challanging due to language ambiguity, synonyms, polysemies and so on. This problem is exacerbated when the queries are out-of-vocabulary (OOV), i.e., having no equivalent matchings in the ontology. In this work, we focus on the problem of hierarchical concept retrieval from SNOMED CT with OOV queries, and propose an approach based on language model-based ontology embeddings. For evaluation, we construct OOV queries annotated against SNOMED CT concepts, testing the retrieval of the most direct subsumers and their less relevant ancestors. We find that our method outperforms the baselines including SBERT and two lexical matching methods. While evaluated against SNOMED CT, the approach is generalisable and can be extended to other ontologies. We release code, tools, and evaluation datasets at [https://github.com/jonathondilworth](HR-OOV).

## Usage

## Features

## Reproducibility

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
