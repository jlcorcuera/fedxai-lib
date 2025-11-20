# fedxai-lib

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

**fedxai-lib** is a Python library for Federated Learning (FL) of eXplainable Artificial Intelligence (XAI) models. The library provides privacy-preserving implementations of interpretable machine learning algorithms, enabling distributed training while maintaining data privacy and model transparency.

The current version of the framework includes the implementation of federated clustering algorithms (Fuzzy C-Means and C-Means for both horizontal and vertical data partitioning) [[3]](#3), a federated Fuzzy Regression Tree (FRT) algorithm for interpretable regression tasks [[2]](#2), a federated Rule-Based Classifier (FRBC) for explainable classification [[4]](#4), and a federated SHAP implementation for consistent post-hoc explainability [[5]](#5). These algorithms are designed to operate in distributed environments where data cannot be centralized due to privacy, regulatory, or operational constraints.

This work has been developed by the [Artificial Intelligence R&D Group](https://ai.dii.unipi.it/) at the Department of Information Engineering, University of Pisa. fedxai-lib has supported research, development, and demonstration activities concerning the FL of XAI models.

<p align="center">
	<img src="./images/logo-DII.png" alt="Department of Information Engineering" style="height: 80px">
	&emsp;&emsp;
	<img src="./images/logo_fair.png" alt="FAIR Project" style="height: 80px">
</p>

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Local Federation Execution](#local-federation-execution)
  - [Docker-Based Distributed Federation](#docker-based-distributed-federation)
- [Algorithm Hyperparameters](#algorithm-hyperparameters)
- [License](#license)
- [Contributors](#contributors)
- [Citations](#citations)
- [Acknowledgments](#acknowledgments)

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
│   │   │   ├── federated_frt/                 # Fuzzy Regression Tree
│   │   │   ├── federated_frbc/                # Rule-Based Classifier
│   │   │   └── federated_shap/                # Federated SHAP
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

- `fedlang-py == 0.0.1`
- `pandas >= 2.3.3`
- `numpy >= 2.3.4`
- `numba >= 0.62.1`
- `scikit-learn >= 1.7.2`
- `simpful >= 2.12.0`
- `shap >= 0.46.0`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/jlcorcuera/fedxai-lib.git
cd fedxai-lib

# Install dependencies
poetry install
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
poetry run python tests/test_fed_rbc_rmi_demo_fedxai_lib.py
poetry run python tests/test_fed_shap_rmi.py
```

Additional examples for all implemented algorithms can be found in the [src/tests/](src/tests/) directory.

### Docker-Based Distributed Federation

For realistic federated scenarios with Docker containers distributed across multiple machines, please refer to the comprehensive [Illustrative Example](Illustrative_Example.md) which provides:

- Step-by-step Docker infrastructure setup
- Environment configuration guidelines
- Federation execution and monitoring
- Troubleshooting common issues

**Quick Start:**

```bash
# Build Docker images
docker build --progress=plain -f Dockerfile.fedxai_lib -t fedxai .
docker build --progress=plain -f Dockerfile.requester -t fedlang-requester .

# Launch federation infrastructure
# IMPORTANT: Start the director first, then the clients
docker compose -f docker-compose-director.yml up -d  # On director machine (start first)
docker compose -f docker-compose-clients.yml up -d   # On client machines (start after director)
docker compose -f docker-compose-requester.yml up -d  # On requester machine

# Execute federation
docker exec -it requester /bin/bash
cd scripts
./run_federation.sh ./executions/federated_frt_weather_izimir.json
```

For detailed instructions, configuration examples, and troubleshooting, see **[Illustrative_Example.md](Illustrative_Example.md)**.

---

## Algorithm Hyperparameters

Detailed documentation of all hyperparameters for each implemented algorithm is available in:

**[Algorithm_Hyperparameters.md](Algorithm_Hyperparameters.md)**

This reference provides:

- Complete hyperparameter descriptions for all algorithms
- Parameter types, default values, and valid ranges
- Usage examples and best practices
- Privacy-related parameter configurations

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0.

---

## Contributors

- **Alessio Bechini** - [Google Scholar](https://scholar.google.it/citations?user=ooYOGP4AAAAJ&hl=it&oi=ao) - [alessio.bechini@unipi.it](mailto:alessio.bechini@unipi.it)
- **José Luis Corcuera Bárcena** - [Google Scholar](https://scholar.google.com/) - [jose.corcuera@ing.unipi.it](mailto:jose.corcuera@ing.unipi.it)
- **Mattia Daole** - [Google Scholar](https://scholar.google.it/citations?user=yNletAoAAAAJ&hl=it&oi=ao) - [mattia.daole@phd.unipi.it](mailto:mattia.daole@phd.unipi.it)
- **Pietro Ducange** - [Google Scholar](https://scholar.google.com/) - [pietro.ducange@unipi.it](mailto:pietro.ducange@unipi.it)
- **Francesco Marcelloni** - [Google Scholar](https://scholar.google.com/) - [francesco.marcelloni@unipi.it](mailto:francesco.marcelloni@unipi.it)
- **Giustino Claudio Miglionico** - [Google Scholar](https://scholar.google.it/citations?user=GSRVwE4AAAAJ&hl=it&oi=ao) - [giustino.miglionico@phd.unipi.it](mailto:giustino.miglionico@phd.unipi.it)

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

### <a name="4"></a>[4] Federated Rule-Based Classifier (FRBC)

```bibtex
@inproceedings{daole2024trustworthy,
  title={Trustworthy AI in heterogeneous settings: federated learning of explainable classifiers},
  author={Daole, M. and Ducange, P. and Marcelloni, F. and Renda, A.},
  booktitle={2024 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)},
  pages={1--9},
  year={2024},
  organization={IEEE}
}
```

### <a name="5"></a>[5] Federated SHAP: Consistent Post-hoc Explainability

```bibtex
@inproceedings{ducange2024consistent,
  title={Consistent post-hoc explainability in federated learning through federated fuzzy clustering},
  author={Ducange, Pietro and Marcelloni, Francesco and Renda, Alessandro and Ruffini, Fabrizio},
  booktitle={2024 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)},
  pages={1--10},
  year={2024},
  organization={IEEE}
}
```

---

## Acknowledgments

This work has been developed by the **Artificial Intelligence R&D Group** at the Department of Information Engineering, University of Pisa.

fedxai-lib has supported research, development, and demonstration activities concerning the FL of XAI models. This work has been funded by:

- **Bando FAIR Trasferimento Tecnologico**: "Sviluppo di una libreria modulare per l'apprendimento di modelli di Explainable Artificial Intelligence in ambienti di Federated Learning" (Development of a modular library for learning Explainable Artificial Intelligence models in Federated Learning environments)

We would like to acknowledge the invaluable support of **Professors Nicola Tonellotto, Tiberio Uricchio, and Alberto Landi**, members of the Fed-XAI Library project working group.

---

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/jlcorcuera/fedxai-lib) or contact the contributors directly.
