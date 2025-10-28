# Docker Guide

This guide shows how to deploy and run federations in Docker using the provided images and compose files.

## Prerequisites

- Docker Engine and Docker Compose

## Build Images

Build core images from the project root:

```
docker build --progress=plain -f Dockerfile.fedxai -t fedxai .
docker build --progress=plain -f Dockerfile.requester -t fedlang-requester .
docker build --progress=plain -f Dockerfile.fedxai_frbc -t fedxai-frbc .
```

## Compose Files

This repo provides compose definitions for a typical setup:

- `docker-compose-director.yml` — Director node (coordinates the federation)
- `docker-compose-clients_frbc.yml` — FRBC client nodes (mount per-client data)
- `docker-compose-requester.yml` — Requester node (submits federation plans)

Launch the stack:

```
docker compose \
  -f docker-compose-director.yml \
  -f docker-compose-clients_frbc.yml \
  -f docker-compose-requester.yml up
```

## Datasets and Volumes

Each client container should mount its dataset partition to the paths expected by the plan file (e.g., `/dataset/X_train.csv`, `/dataset/y_train.csv`, `/dataset/X_test.csv`, `/dataset/y_test.csv`). Also mount a shared `/models` directory to persist trained models.

Example volume section for a client service:

```yaml
volumes:
  - ./datasets/RMI_demo/preprocessed/split_train_0.csv:/dataset/X_train.csv
  - ./datasets/RMI_demo/preprocessed/split_train_0.csv:/dataset/y_train.csv
  - ./datasets/RMI_demo/preprocessed/test.csv:/dataset/X_test.csv
  - ./datasets/RMI_demo/preprocessed/test.csv:/dataset/y_test.csv
  - ./models:/models
```

Adjust according to your dataset splits and plan parameters.

## Run a Plan from the Requester

Attach to the requester container and execute a plan:

```
docker exec -it requester /bin/bash
cd /app/src/scripts
./run_federation.sh ../experiments/plan_federated_frbc_RMI_demo.json
```

Ensure the plan matches the algorithm and dataset paths used by your clients. Example schemas are in `src/fedxai_lib/descriptors/definitions/`.

