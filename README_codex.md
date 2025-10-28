# FedXAI-lib

FedXAI-lib is a library for Federated Learning (FL) of explainable-by-design models. It provides components and examples to run federations locally or in Docker, with a focus on fuzzy and tree-based XAI models.

## Table of Contents

## Table of Contents

- [FedXAI-lib](#fedxai-lib)
  - [Table of Contents](#table-of-contents)
  - [Table of Contents](#table-of-contents-1)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Illustrative Example](#illustrative-example)
- [Setup and Run a Federation](#setup-and-run-a-federation)
  - [Docker Deployment](#docker-deployment)
  - [Further Documentation](#further-documentation)
  - [License](#license)


# Repository Structure

```
├── datasets/                               # Example datasets and splits
│   └── RMI_demo/preprocessed               # RMI demo data (train/test splits)
├── src/
│   ├── fedxai_lib/                         # Library source code
│   │   ├── algorithms/
│   │   │   ├── federated_frbc/             # Federated Fuzzy Rule-Based Classifier
│   │   │   ├── federated_frt/              # Federated Fuzzy Regression Tree
│   │   │   ├── federated_cmeans_horizontal # Federated C-Means (Horizontal)
│   │   │   ├── federated_cmeans_vertical   # Federated C-Means (Vertical)
│   │   │   ├── federated_fcmeans_horizontal# Federated Fuzzy C-Means (Horizontal)
│   │   │   └── federated_fcmeans_vertical  # Federated Fuzzy C-Means (Vertical)
│   │   └── descriptors/                    # Plans and plan loader
│   │       └── definitions/                # Algorithm execution plan schemas
│   ├── scripts/
│   │   ├── run_federation.sh               # Helper to trigger a federation
│   │   └── experiments/                    # Example federation plans
│   └── tests/                              # Local simulations and tests
├── docker-compose-*.yml                    # Compose files for director/clients/requester
├── Dockerfile.*                            # Dockerfiles for images
├── pyproject.toml                          # Project configuration (Poetry)
├── poetry.lock
├── LICENSE
└── README.md
```

# Prerequisites

- [Python 3.11+](https://www.python.org/downloads/release/python-3110/)
- [Poetry](https://python-poetry.org/docs/) - Python dependency and packaging manager
- [Docker](https://docs.docker.com/engine/install/) - Optional, for deployment in containers

Install the environment:

```
poetry install
```



# Illustrative Example

The repository includes ready-to-run examples of federations and unit-test style scripts to simulate distributed training locally. For instance, a fuzzy regression tree (FRT) demo on the Weather Izimir dataset and a fuzzy rule-based classifier (FRBC) demo on the RMI dataset are provided under `src/tests/` and `src/scripts/experiments/`.

Key examples:

- Local simulation tests: see `src/tests/test_fed_frt_weather_izimir.py`, `src/tests/test_frbc_RMI_DEMO.py`, and `src/tests/test_frbc_RMI_DEMO_fedxai_lib.py`.
- Example execution plans: see `src/scripts/experiments/plan_federated_frbc_RMI_demo.json` and other plans in `src/fedxai_lib/descriptors/definitions/`.

Check the [illustrative example][IllustrativeExample] for a step-by-step guide to the usage of Fedxai-lib for learning a Regression Tree in a federated fashion. 


# Setup and Run a Federation

Run locally with Poetry (example: FRT Weather Izimir):

```
cd src
poetry run python tests/test_fed_frt_weather_izimir.py
```

Run an FRBC RMI demo locally (adapt as needed):

```
cd src
poetry run python tests/test_frbc_RMI_DEMO.py
```

To execute a federation based on a plan file from the requester node (see next section for Docker):

```
cd src/scripts
./run_federation.sh ../experiments/plan_federated_frbc_RMI_demo.json
```

## Docker Deployment

Build images:

```
docker build --progress=plain -f Dockerfile.fedxai -t fedxai .
docker build --progress=plain -f Dockerfile.requester -t fedlang-requester .
docker build --progress=plain -f Dockerfile.fedxai_frbc -t fedxai-frbc .
```

Compose files provided (see DOCKER.md for details):

- `docker-compose-director.yml` — Director node
- `docker-compose-clients_frbc.yml` — FRBC clients (mount per-client datasets)
- `docker-compose-requester.yml` — Requester to trigger runs

Launch the stack (example):

```
docker compose -f docker-compose-director.yml -f docker-compose-clients_frbc.yml -f docker-compose-requester.yml up
```

From the requester container, run a plan:

```
docker exec -it requester /bin/bash
cd /app/src/scripts
./run_federation.sh ../experiments/plan_federated_frbc_RMI_demo.json
```

Ensure each client mounts its dataset partition to the expected paths used by the plan (e.g., `/dataset/X_train.csv`, `/dataset/y_train.csv`, etc.) and a shared models directory (e.g., `/models`).

## Further Documentation

- Usage Guide: `USAGE.md`
- Docker Guide: `DOCKER.md`

## License

Licensed under the Apache License 2.0. See `LICENSE`.








[IllustrativeExample]: Illustrative_Example.md