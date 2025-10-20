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
│   ├── ...
tests/                             # Unit tests and local federation simulations
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
from fedxai_lib import run_fedxai_experiment, FedXAIAlgorithm
from fedxai_lib.algorithms.federated_frt.client import FedFRTClient
from fedxai_lib.algorithms.federated_frt.server import FedFRTServer

parameters = {
      "gain_threshold": 0.0001,
      "max_number_rounds": 100,
      "num_fuzzy_sets": 5,
      "max_depth": None,
      "min_samples_split_ratio": 0.1,
      "min_num_clients": 20,
      "obfuscate": True,
      "features_names": ["Max_temperature","Min_temperature","Dewpoint","Precipitation","Sea_level_pressure","Standard_pressure","Visibility","Wind_speed","Max_wind_speed"],
      "target": "Mean_temperature",
      "dataset_X_train": "/dataset/X_train.csv",
      "dataset_y_train": "/dataset/y_train.csv",
      "dataset_X_test": "/dataset/X_test.csv",
      "dataset_y_test": "/dataset/y_test.csv",
      "model_output_file": "/models/frt_weather_izimir.pickle"
}

clients = [FedFRTClient(type='client', id = idx, scaler_X=scaler_x, scaler_y=scaler_y,
                        X_train=dataset_by_client.get(idx)['X_train'],
                        y_train=dataset_by_client.get(idx)['y_train'],
                        X_test=dataset_by_client.get(idx)['X_test'],
                        y_test=dataset_by_client.get(idx)['y_test']) for idx in range(num_clients)]

server = FedFRTServer(type='server')

run_fedxai_experiment(FedXAIAlgorithm.FED_FRT_HORIZONTAL, server, clients, parameters)
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
$ ./run_federation.sh ../executions/federated_frt_weather_izimir.json
```

where the content of federated_frt_weather_izimir.json is:

```json
{
    "algorithm": "federated_frt",
    "parameters": {
      "gain_threshold": 0.0001,
      "max_number_rounds": 100,
      "num_fuzzy_sets": 5,
      "max_depth": null,
      "min_samples_split_ratio": 0.1,
      "min_num_clients": 20,
      "obfuscate": true,
      "features_names": ["Max_temperature","Min_temperature","Dewpoint","Precipitation","Sea_level_pressure","Standard_pressure","Visibility","Wind_speed","Max_wind_speed"],
      "target": "Mean_temperature",
      "dataset_X_train": "/dataset/X_train.csv",
      "dataset_y_train": "/dataset/y_train.csv",
      "dataset_X_test": "/dataset/X_test.csv",
      "dataset_y_test": "/dataset/y_test.csv",
      "model_output_file": "/models/frt_weather_izimir.pickle"
  }
}
```

---

## 🔭 Future Work

- Support for **Federated Neural Networks**
- Automated metrics and visualization tools

---

## 📜 License

Licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
