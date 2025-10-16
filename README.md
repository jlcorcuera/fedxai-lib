# FedXAI-lib · <img src="https://avatars.githubusercontent.com/u/153393?s=48&v=4" style="background:white;" alt="Erlang" width="28"/> + <img src="https://www.vectorlogo.zone/logos/python/python-icon.svg" alt="Python" width="28"/>  
*A Federated Learning of Explainable Artificial Intelligence library

**FedXAI-lib** is a TBU.

---

## 🧮 Supported Algorithms

FedLang-Py currently implements the following Federated Learning algorithms:

- **Federated Fuzzy C-Means** (Horizontal Partitioning)  
- **Federated C-Means** (Horizontal Partitioning)  
- **Federated Fuzzy C-Means** (Vertical Partitioning)  
- **Federated Fuzzy Regression Tree (FRT)**  

---

## 🧩 Requirements

Before running the framework, ensure that the following dependencies are installed:

- [Python 3.11+](https://www.python.org/downloads/)  
- [Poetry](https://python-poetry.org/docs/#installation)

---

## 📁 Project Structure

FedXAI-lib follows a modular and extensible structure that allows Data Scientists to easily integrate and test new federated algorithms.

```bash
fedxai_lib/
├── algorithms/                        # Federated algorithm implementations
│   ├── federated_fc_means_horizontal  # Fuzzy C-Means (Horizontal)
│   ├── federated_frt                  # Fuzzy Regression Tree (FRT)
├── plans/                             # JSON-based Federation Plans
├── tests/                             # Unit tests and local federation simulations
│   ├── test_fed_fcmeans_xclara.py
│   └── test_fed_frt_weather_izimir.py
├── pyproject.toml
└── README.md
```

---


## 🧪 Running Federations Locally

You can execute federated algorithms locally for testing and debugging.

**Example:**

```python
from fedlangpy.algorithms.federated_cmeans_vertical.client import FederatedVerticalCMClient
from fedlangpy.algorithms.federated_cmeans_vertical.server import FederatedVerticalCMServer
from fedlangpy.core.utils import load_plan, run_experiment

plan_path = './plans/plan_federated_cmeans_vertical_xclara.json'
fl_plan = load_plan(plan_path, server_class=FederatedVerticalCMServer, client_class=FederatedVerticalCMClient)

clients = [FederatedVerticalCMClient(type='client', id=idx, dataset=dataset_chunks[idx]) for idx in range(num_clients)]
server = FederatedVerticalCMServer(type='server')

run_experiment(fl_plan, server, clients)
```

### Environment Setup

```bash
$ poetry install
$ cd src
```

### Run Example Federation

```bash
(fedlangpy) $ poetry run python tests/test_fed_frt_weather_izimir.py
```

---

## 🐳 Running Federations in Docker

You can deploy complete federations using Docker containers to simulate distributed environments.

### 1️⃣ Build Docker Images

```bash
$ docker build --progress=plain -f Dockerfile.fedxai -t fedxai .
$ docker build --progress=plain -f Dockerfile.requester -t fedlang-requester .
```

---

### 2️⃣ Define `docker-compose.yml`

Create a `docker-compose.yml` defining the **Director**, **Clients**, and **Requester** services.

Required environment variables:

| Variable                | Description                             |
|-------------------------|-----------------------------------------|
| `FEDLANG_NODE_TYPE`     | Node type: `director` or `client`       |
| `FEDLANG_NODE_NAME`     | Node name                               |
| `FEDLANG_COOKIE`        | Erlang cookie for distributed messaging |
| `FEDLANG_DIRECTOR_NAME` | Director node name                      |

Each client container must mount its local dataset partition, e.g.:

```yaml
volumes:
      - ./datasets_splits/WeatherIzimir/client_X_train_1.csv:/dataset/X_train.csv
      - ./datasets_splits/WeatherIzimir/client_y_train_1.csv:/dataset/y_train.csv
      - ./datasets_splits/WeatherIzimir/client_X_test_1.csv:/dataset/X_test.csv
      - ./datasets_splits/WeatherIzimir/client_y_test_1.csv:/dataset/y_test.csv
      - /models:/models
```

Then launch the federation environment:

```bash
$ docker compose up
```

---

### 3️⃣ Run the Federation

Access the **Requester** node:

```bash
$ docker exec -it requester /bin/bash
```

Run the federation by sending a plan execution request to the Director:

```bash
$ cd scripts
$ ./run_federation.sh ../plans/plan_federated_frt_weather_izimir.json
```

---

## 🔭 Future Work

- Support for **Federated Neural Networks**
- Automated metrics and visualization tools

---

## 📜 License

Licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
