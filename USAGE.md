# Usage Guide

This guide covers running FedXAI-lib locally for quick experiments and simulations.

## Prerequisites

- Python 3.11+
- Poetry

Install dependencies:

```
poetry install
```

Enter the source folder for running tests/examples:

```
cd src
```

## Run Local Simulations

Local simulations run a full federation within a single process, useful for development and debugging.

### Fuzzy Regression Tree (FRT) — Weather Izimir

```
poetry run python tests/test_fed_frt_weather_izimir.py
```

### Fuzzy Rule-Based Classifier (FRBC) — RMI Demo

```
poetry run python tests/test_frbc_RMI_DEMO.py
```

or the variant using the library entry points:

```
poetry run python tests/test_frbc_RMI_DEMO_fedxai_lib.py
```

## Plans and Descriptors

- High-level plan schema definitions: `src/fedxai_lib/descriptors/definitions/`
- Ready-to-run example plan: `src/scripts/experiments/plan_federated_frbc_RMI_demo.json`

You can trigger a plan using the helper script (outside of Docker) via:

``+
cd src/scripts
./run_federation.sh ../experiments/plan_federated_frbc_RMI_demo.json
``+

Adjust dataset paths in the plan to match your local environment.

