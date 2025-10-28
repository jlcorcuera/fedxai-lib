# fedxai-lib

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

**fedxai-lib** is a Python library for Federated Learning (FL) of eXplainable Artificial Intelligence (XAI) models. The library provides privacy-preserving implementations of interpretable machine learning algorithms, enabling distributed training while maintaining data privacy and model transparency.

This work has been developed by the [Artificial Intelligence R&D Group](https://ai.dii.unipi.it/) at the Department of Information Engineering, University of Pisa. fedxai-lib has supported research, development, and demonstration activities concerning the FL of XAI models.

<p align="center">
	<img src="./images/logo-DII.png" alt="Department of Information Engineering" style="height: 80px">
	&emsp;&emsp;
	<img src="./images/logo_fair.png" alt="FAIR Project" style="height: 80px">
</p>

---

## Table of Contents

- [Implemented Algorithms](#implemented-algorithms)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Local Federation Execution](#local-federation-execution)
  - [Docker-Based Distributed Federation](#docker-based-distributed-federation)
- [Illustrative Example](#illustrative-example)
- [License](#license)
- [Contributors](#contributors)
- [Citations](#citations)
- [Acknowledgments](#acknowledgments)

---

## Implemented Algorithms

fedxai-lib currently implements the following Federated Learning algorithms for XAI:

### Clustering Algorithms

- **Federated Fuzzy C-Means** (Horizontal Partitioning) [[3]](#3)
- **Federated C-Means** (Horizontal Partitioning) [[3]](#3)
- **Federated Fuzzy C-Means** (Vertical Partitioning) [[3]](#3)
- **Federated C-Means** (Vertical Partitioning) [[3]](#3)

### Tree-Based Algorithms

- **Federated Fuzzy Regression Tree (FRT)** [[2]](#2)
  - Privacy-preserving fuzzy regression tree construction
  - Interpretable rules with fuzzy set-based splits
  - Obfuscation mechanisms to prevent inference attacks

All algorithms are built on top of the **fedlang** middleware [[1]](#1), which provides actor-based distributed execution support for federated learning experiments.

---

## Repository Structure

```
fedxai-lib/
├── src/
│   ├── fedxai_lib/                     # Main library package
│   │   ├── algorithms/                 # Federated algorithm implementations
│   │   │   ├── federated_fcmeans_horizontal/  # Fuzzy C-Means (Horizontal)
│   │   │   ├── federated_fcmeans_vertical/    # Fuzzy C-Means (Vertical)
│   │   │   ├── federated_cmeans_horizontal/   # C-Means (Horizontal)
│   │   │   ├── federated_cmeans_vertical/     # C-Means (Vertical)
│   │   │   └── federated_frt/                 # Fuzzy Regression Tree
│   │   │       ├── client.py           # Client-side logic
│   │   │       ├── server.py           # Server-side aggregation
│   │   │       ├── model.py            # Model representation
│   │   │       ├── node.py             # Tree node structures
│   │   │       └── utils/              # Algorithm utilities
│   │   ├── descriptors/                # Federated learning plan descriptors
│   │   │   └── definitions/            # JSON-based execution plans
│   │   └── __init__.py                 # Public API
│   ├── tests/                          # Unit tests and local simulations
│   └── scripts/                        # Federation execution scripts
│       ├── run_federation.sh           # Shell script to execute federations
│       └── executions/                 # Example federation configurations
├── datasets/                           # Raw datasets
├── datasets_splits/                    # Dataset partitions for clients
├── models/                             # Trained model outputs
├── dist/                               # Built package distributions
├── Dockerfile.fedxai_lib               # Docker image for clients/director
├── Dockerfile.requester                # Docker image for requester node
├── docker-compose-director.yml         # Director service configuration
├── docker-compose-clients.yml          # Client services configuration
├── docker-compose-requester.yml        # Requester service configuration
├── pyproject.toml                      # Poetry dependencies and metadata
├── requirements.txt                    # pip-compatible requirements
├── LICENSE                             # Apache 2.0 license
└── README.md                           # This file
```

---

## Prerequisites

Before using fedxai-lib, ensure the following dependencies are installed:

- **Python 3.11+** - [Download](https://www.python.org/downloads/)
- **Poetry** - Dependency management tool ([Installation Guide](https://python-poetry.org/docs/#installation))
- **Docker** (optional) - For distributed federation deployments ([Get Docker](https://docs.docker.com/get-docker/))

### Python Package Dependencies

The library depends on the following key packages (automatically installed via Poetry):

- `fedlang-py >= 0.0.7` - Federated learning middleware
- `pandas >= 2.3.3` - Data manipulation
- `numpy >= 2.3.4` - Numerical computations
- `numba >= 0.62.1` - Performance optimization
- `scikit-learn >= 1.7.2` - Machine learning utilities
- `simpful >= 2.12.0` - Fuzzy logic operations

---

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/jlcorcuera/fedxai-lib.git
cd fedxai-lib

# Install dependencies
poetry install
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/jlcorcuera/fedxai-lib.git
cd fedxai-lib

# Install from requirements file
pip install -r requirements.txt
```

---

## Usage

fedxai-lib supports two execution modes: **local federation** (for testing and development) and **distributed federation** (using Docker containers).

### Local Federation Execution

Local execution simulates federated learning by running all clients and server in a single process. This is ideal for algorithm testing and debugging.

**Example: Federated Fuzzy Regression Tree**

```python
from fedxai_lib import run_fedxai_experiment, FedXAIAlgorithm
from fedxai_lib.algorithms.federated_frt.client import FedFRTClient
from fedxai_lib.algorithms.federated_frt.server import FedFRTServer

# Define algorithm parameters
parameters = {
    "gain_threshold": 0.0001,
    "max_number_rounds": 100,
    "num_fuzzy_sets": 5,
    "max_depth": None,
    "min_samples_split_ratio": 0.1,
    "min_num_clients": 20,
    "obfuscate": True,
    "features_names": ["Max_temperature", "Min_temperature", "Dewpoint",
                       "Precipitation", "Sea_level_pressure", "Standard_pressure",
                       "Visibility", "Wind_speed", "Max_wind_speed"],
    "target": "Mean_temperature",
    "dataset_X_train": "/dataset/X_train.csv",
    "dataset_y_train": "/dataset/y_train.csv",
    "dataset_X_test": "/dataset/X_test.csv",
    "dataset_y_test": "/dataset/y_test.csv",
    "model_output_file": "/models/frt_weather_izimir.pickle"
}

# Create clients with their local data partitions
clients = [
    FedFRTClient(type='client', id=idx,
                 scaler_X=scaler_x, scaler_y=scaler_y,
                 X_train=dataset_by_client[idx]['X_train'],
                 y_train=dataset_by_client[idx]['y_train'],
                 X_test=dataset_by_client[idx]['X_test'],
                 y_test=dataset_by_client[idx]['y_test'])
    for idx in range(num_clients)
]

# Create server
server = FedFRTServer(type='server')

# Run federated experiment
run_fedxai_experiment(FedXAIAlgorithm.FED_FRT_HORIZONTAL, server, clients, parameters)
```

**Running test scripts:**

```bash
cd src
poetry run python tests/test_fed_frt_weather_izimir.py
poetry run python tests/test_fed_fcmeans_horizontal_xclara.py
```

### Docker-Based Distributed Federation

For realistic federated scenarios, deploy the system using Docker containers distributed across multiple machines.

**Step 1: Build Docker Images**

```bash
# Build fedxai library image (for clients and director)
docker build --progress=plain -f Dockerfile.fedxai_lib -t fedxai .

# Build requester image (for sending federation requests)
docker build --progress=plain -f Dockerfile.requester -t fedlang-requester .
```

**Step 2: Configure Docker Compose Files**

The repository includes three compose files:

- `docker-compose-director.yml` - Runs the director (server/aggregator) node
- `docker-compose-clients.yml` - Runs client nodes with their data partitions
- `docker-compose-requester.yml` - Runs the requester node to initiate federations

**Key environment variables:**

| Variable                | Description                                      |
|-------------------------|--------------------------------------------------|
| `FEDLANG_NODE_TYPE`     | Node type: `director`, `client`, or `requester` |
| `FEDLANG_NODE_NAME`     | Unique node identifier (format: name@IP)        |
| `FEDLANG_COOKIE`        | Erlang cookie for distributed authentication     |
| `FEDLANG_DIRECTOR_NAME` | Director node identifier for clients to connect  |

**Step 3: Launch Federation Infrastructure**

```bash
# On the director machine
docker compose -f docker-compose-director.yml up -d

# On client machines (adjust IPs and data volumes)
docker compose -f docker-compose-clients.yml up -d

# On the requester machine
docker compose -f docker-compose-requester.yml up -d
```

**Step 4: Execute Federated Learning**

Access the requester container and run a federation:

```bash
# Enter requester container
docker exec -it requester /bin/bash

# Navigate to scripts directory
cd scripts

# Run federation with configuration file
./run_federation.sh ./executions/federated_frt_weather_izimir.json
```

The JSON configuration file specifies the algorithm and parameters:

```json
{
  "algorithm": "federated_frt",
  "parameters": {
    "gain_threshold": 0.0001,
    "max_number_rounds": 100,
    "num_fuzzy_sets": 5,
    "max_depth": null,
    "min_samples_split_ratio": 0.1,
    "obfuscate": true,
    "features_names": ["Max_temperature", "Min_temperature", ...],
    "target": "Mean_temperature",
    "dataset_X_train": "/dataset/X_train.csv",
    "model_output_file": "/models/frt_weather_izimir.pickle"
  }
}
```

---

## Illustrative Example

A comprehensive step-by-step example demonstrating the execution of a Federated Fuzzy Regression Tree using Docker containers is available in:

**[Illustrative_Example.md](Illustrative_Example.md)**

This example covers:
- Dataset preparation and partitioning
- Docker environment setup
- Federation execution and monitoring
- Model evaluation and interpretation

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0.

---

## Contributors

- **Alessio Bechini** - [alessio.bechini@unipi.it](mailto:alessio.bechini@unipi.it)
- **José Luis Corcuera Bárcena** - [Google Scholar](https://scholar.google.com/) - [joseluis.corcuera@phd.unipi.it](mailto:joseluis.corcuera@phd.unipi.it)
- **Mattia Daole** - [mattia.daole@phd.unipi.it](mailto:mattia.daole@phd.unipi.it)
- **Pietro Ducange** - [Google Scholar](https://scholar.google.com/) - [pietro.ducange@unipi.it](mailto:pietro.ducange@unipi.it)
- **Francesco Marcelloni** - [Google Scholar](https://scholar.google.com/) - [francesco.marcelloni@unipi.it](mailto:francesco.marcelloni@unipi.it)
- **Giustino Miglionico** - [giustino.miglionico@phd.unipi.it](mailto:giustino.miglionico@phd.unipi.it)

---

## Citations

If you use fedxai-lib in your research, please cite the relevant papers:

### <a name="1"></a>[1] Middleware Support for Federated Learning

```bibtex
@article{bechini2025devising,
  title={Devising an actor-based middleware support to federated learning experiments and systems},
  author={Bechini, Alessio and Corcuera Barcena, Jose Luis},
  journal={Future Generation Computer Systems},
  volume={166},
  pages={107646},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.future.2024.107646}
}
```

### <a name="2"></a>[2] Federated Fuzzy Regression Trees

```bibtex
@article{barcena2025increasing,
  title={Increasing trust in AI through privacy preservation and model explainability: Federated Learning of Fuzzy Regression Trees},
  author={B{\'a}rcena, Jos{\'e} Luis Corcuera and Ducange, Pietro and Marcelloni, Francesco and Renda, Alessandro},
  journal={Information Fusion},
  volume={113},
  pages={102598},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.inffus.2024.102598}
}
```

### <a name="3"></a>[3] Federated C-Means and Fuzzy C-Means Clustering

```bibtex
@article{federated_cmeans,
  title={Federated C-Means and Fuzzy C-Means Clustering Algorithms for Horizontally and Vertically Partitioned Data},
  author={[Authors to be specified]},
  journal={[To be published]},
  year={2025}
}
```

---

## Acknowledgments

This work has been developed by the **Artificial Intelligence R&D Group** at the Department of Information Engineering, University of Pisa.

fedxai-lib has supported research, development, and demonstration activities concerning the FL of XAI models. This work has been funded by:

- The **PNRR project - M4C2 - Investimento 1.3**, Partenariato Esteso PE00000013 - **FAIR - Future Artificial Intelligence Research** - Spoke 1 "Human-centered AI"
- The **Italian Ministry of University and Research (MUR)** in the framework of the **FoReLab** and **CrossLab** projects (Departments of Excellence)

---

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/jlcorcuera/fedxai-lib) or contact the contributors directly.
