# Illustrative Example: Federated Fuzzy Regression Tree

This document provides a comprehensive step-by-step guide to executing a **Federated Fuzzy Regression Tree (FRT)** experiment using fedxai-lib in a distributed Docker environment. The example demonstrates training an interpretable regression model on weather data distributed across multiple client nodes.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Environment Setup](#environment-setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Docker Infrastructure Deployment](#docker-infrastructure-deployment)
6. [Federation Execution](#federation-execution)
7. [Results and Model Interpretation](#results-and-model-interpretation)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Scenario

We will train a Federated Fuzzy Regression Tree to predict **mean temperature** based on various meteorological features from the **Weather Izimir** dataset. The dataset is partitioned horizontally across multiple client nodes, simulating a realistic federated learning scenario where data cannot leave its source.

### Architecture

The federation consists of three types of nodes:

- **Director** (1 node): Orchestrates the federation and aggregates client statistics
- **Clients** (4 nodes): Each holds a private partition of the training/test data
- **Requester** (1 node): Initiates federation requests and monitors execution

### Learning Objective

Build an interpretable fuzzy regression tree that:
- Maintains data privacy (raw data never leaves client nodes)
- Provides explainable predictions through fuzzy rules
- Implements obfuscation to prevent inference attacks

---

## Dataset Description

### Weather Izimir Dataset

The dataset contains meteorological measurements from Izimir, Turkey, with the following features:

**Input Features:**
- `Max_temperature` - Maximum daily temperature (°C)
- `Min_temperature` - Minimum daily temperature (°C)
- `Dewpoint` - Dew point temperature (°C)
- `Precipitation` - Precipitation amount (mm)
- `Sea_level_pressure` - Sea level atmospheric pressure (hPa)
- `Standard_pressure` - Standard atmospheric pressure (hPa)
- `Visibility` - Visibility distance (km)
- `Wind_speed` - Average wind speed (km/h)
- `Max_wind_speed` - Maximum wind speed (km/h)

**Target Variable:**
- `Mean_temperature` - Mean daily temperature (°C)

### Data Partitioning

The dataset is partitioned horizontally into 5 client datasets using K-Fold cross-validation:

```
datasets_splits/WeatherIzimir/
├── client_X_train_1.csv
├── client_y_train_1.csv
├── client_X_test_1.csv
├── client_y_test_1.csv
├── client_X_train_2.csv
├── client_y_train_2.csv
├── ... (clients 3, 4, 5)
```

Each client receives approximately 20% of the data, ensuring IID (Independent and Identically Distributed) partitioning.

---

## Environment Setup

### Prerequisites

Ensure you have the following installed:

- Docker (version 20.10+)
- Docker Compose (version 1.29+)
- Git

### Directory Structure

```bash
fedxai-lib/
├── datasets_splits/WeatherIzimir/   # Client data partitions
├── models/                          # Output directory for trained models
├── Dockerfile.fedxai_lib            # Docker image for fedxai library
├── Dockerfile.requester             # Docker image for requester
├── docker-compose-director.yml      # Director configuration
├── docker-compose-clients.yml       # Clients configuration
└── docker-compose-requester.yml     # Requester configuration
```

### Build Docker Images

For realistic federated scenarios, deploy the system using Docker containers distributed across multiple machines.

```bash
# Navigate to repository root
cd fedxai-lib

# Build fedxai library image (used by director and clients)
docker build --progress=plain -f Dockerfile.fedxai_lib -t fedxai .

# Build requester image
docker build --progress=plain -f Dockerfile.requester -t fedlang-requester .
```

**Expected Output:**
```
[+] Building 45.3s (12/12) FINISHED
Successfully tagged fedxai:latest
Successfully tagged fedlang-requester:latest
```

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

---

## Dataset Preparation

If you don't already have the dataset partitions, generate them using the test script:

```bash
cd src
poetry run python tests/test_fed_frt_weather_izimir.py
```

This script:
1. Loads the raw `WeatherIzimir.csv` dataset
2. Normalizes features using RobustScaler and MinMaxScaler
3. Partitions data into 5 client datasets using K-Fold
4. Saves partitions to `datasets_splits/WeatherIzimir/`

**Verify data partitions exist:**

```bash
ls -lh datasets_splits/WeatherIzimir/
```

**Expected Output:**
```
client_X_train_1.csv
client_y_train_1.csv
client_X_test_1.csv
client_y_test_1.csv
... (20 files total for 5 clients)
```

---

## Docker Infrastructure Deployment

### Network Architecture

The federation uses the following network configuration:

- **Director**: `172.20.0.2:9000` (bridge network)
- **Clients**: Host network mode with ports `9001-9004`
- **Requester**: `172.20.0.254` (bridge network)

### Step 1: Configure Environment Variables

Edit the docker-compose files to match your network setup:

**docker-compose-director.yml:**
```yaml
environment:
  - FEDLANG_NODE_NAME=director@172.16.2.185  # Director's external IP
  - FEDLANG_COOKIE=050df51e-1c75-433e-95c3-8e3e1926d6a6
```

**docker-compose-clients.yml:**
```yaml
environment:
  - FEDLANG_NODE_NAME=client1@172.16.6.42  # Client host IP
  - FEDLANG_DIRECTOR_NAME=director@172.16.2.185  # Director's external IP
  - FEDLANG_COOKIE=050df51e-1c75-433e-95c3-8e3e1926d6a6
```

**docker-compose-requester.yml:**
```yaml
environment:
  - FEDLANG_DIRECTOR_NAME=director@172.16.2.185  # Director's external IP
  - ERL_FLAGS: -setcookie 050df51e-1c75-433e-95c3-8e3e1926d6a6
```

**Important:**
- Replace IP addresses with your actual machine IPs
- Keep the same `FEDLANG_COOKIE` value across all nodes for authentication

### Step 2: Create Shared Model Directory

```bash
# Create directory for model outputs (shared across containers)
sudo mkdir -p /models
sudo chmod 777 /models
```

### Step 3: Launch Director Node

On the **director machine**:

```bash
docker compose -f docker-compose-director.yml up -d
```

**Verify director is running:**

```bash
docker logs director
```

**Expected Output:**
```
Starting Erlang node: director@172.16.2.185
FedLang Director initialized successfully
Waiting for client connections...
```

### Step 4: Launch Client Nodes

On **client machine(s)**:

```bash
docker compose -f docker-compose-clients.yml up -d
```

**Verify clients are running:**

```bash
docker ps
docker logs client1
docker logs client2
docker logs client3
docker logs client4
```

**Expected Output (per client):**
```
Starting Erlang node: client1@172.16.6.42
Connected to director: director@172.16.2.185
FedLang Client ready for federations
```

### Step 5: Launch Requester Node

On the **requester machine**:

```bash
docker compose -f docker-compose-requester.yml up -d
```

**Verify requester is running:**

```bash
docker logs requester
```

---

## Federation Execution

### Step 1: Access Requester Container

```bash
docker exec -it requester /bin/bash
```

You are now inside the requester container shell.

### Step 2: Prepare Federation Configuration

Navigate to the scripts directory:

```bash
cd scripts
ls executions/
```

**View the federation configuration:**

```bash
cat executions/federated_frt_weather_izimir.json
```

**Configuration Details:**

```json
{
  "algorithm": "federated_frt",
  "parameters": {
    "gain_threshold": 0.0001,           // Minimum information gain for splits
    "max_number_rounds": 100,           // Maximum tree depth (rounds)
    "num_fuzzy_sets": 5,                // Number of fuzzy sets per feature
    "max_depth": null,                  // No depth limit
    "min_samples_split_ratio": 0.1,     // Min 10% of samples to split
    "min_num_clients": 20,              // Minimum samples for privacy
    "obfuscate": true,                  // Enable privacy obfuscation
    "features_names": [                 // Input feature names
      "Max_temperature", "Min_temperature", "Dewpoint",
      "Precipitation", "Sea_level_pressure", "Standard_pressure",
      "Visibility", "Wind_speed", "Max_wind_speed"
    ],
    "target": "Mean_temperature",       // Target variable
    "dataset_X_train": "/dataset/X_train.csv",
    "dataset_y_train": "/dataset/y_train.csv",
    "dataset_X_test": "/dataset/X_test.csv",
    "dataset_y_test": "/dataset/y_test.csv",
    "model_output_file": "/models/frt_weather_izimir.pickle"
  }
}
```

### Step 3: Execute Federation

Run the federation script:

```bash
./run_federation.sh ./executions/federated_frt_weather_izimir.json
```

### Step 4: Monitor Execution

The federation execution proceeds through the following stages:

**Stage 1: Initialization**
```
[Stage 1/7] Initialization of parties
  -> Server initialization
  -> Client initialization (4 clients)
```

**Stage 2: Tree Initialization**
```
[Stage 2/7] Initializing tree
  -> Clients compute root node statistics (WSS, WLS, WS)
  -> Server initializes root node with variance
```

**Stage 3: Tree Growing** (Iterative)
```
[Stage 3/7] Grow tree (Round 1)
  -> Server selects features to evaluate
  -> Clients compute fuzzy set statistics per feature
  -> Server selects best feature split (max information gain)
  -> Server grows tree nodes

[Stage 3/7] Grow tree (Round 2)
  -> Process repeats for new tree level
  ...
```

**Stage 4: Model Building**
```
[Stage 4/7] Initialize tree model
  -> Server extracts fuzzy rules from tree structure
```

**Stage 5: Consequent Computation**
```
[Stage 5/7] Computing consequents
  -> Clients compute weighted least squares matrices
  -> Server aggregates and solves for consequent parameters
```

**Stage 6: Rule Weight Computation**
```
[Stage 6/7] Computing rules' weights
  -> Clients compute firing strengths and errors
  -> Server computes fuzzy confidence and support
  -> Server calculates final rule weights
```

**Stage 7: Model Saving**
```
[Stage 7/7] Save tree model
  -> Server saves global model to /models/frt_weather_izimir.pickle
  -> Clients save personalized models (optional)

Federation completed successfully!
```

### Step 5: Retrieve Trained Model

Exit the requester container:

```bash
exit
```

The trained model is now available on the host machine:

```bash
ls -lh /models/
```

**Expected Output:**
```
frt_weather_izimir.pickle        # Global federated model
frt_weather_izimir_client_0.pickle
frt_weather_izimir_client_1.pickle
... (client-specific models)
```

---

## Results and Model Interpretation

### Loading the Trained Model

```python
import pickle

# Load the trained federated model
with open('/models/frt_weather_izimir.pickle', 'rb') as f:
    model = pickle.load(f)

# View model structure
print(f"Number of fuzzy rules: {len(model.get_rules())}")
print(f"Tree depth: {model.root_node.depth}")
```

### Understanding Fuzzy Rules

The FRT model consists of interpretable fuzzy rules in the form:

```
IF Max_temperature is HIGH AND Dewpoint is MEDIUM
THEN Mean_temperature = 0.85 * Max_temperature + 0.12 * Dewpoint + 0.03
     (Weight: 0.92, Support: 0.15, Confidence: 0.88)
```

### Making Predictions

```python
import numpy as np

# Example input: [Max_temp, Min_temp, Dewpoint, Precip, Sea_press, Std_press, Vis, Wind, Max_wind]
input_sample = np.array([[25.5, 18.2, 15.3, 0.0, 1013.2, 1012.8, 10.0, 12.5, 20.3]])

# Predict
prediction, rule_id, num_active_rules = model.predict(input_sample)[0]

print(f"Predicted Mean Temperature: {prediction:.2f}°C")
print(f"Activated Rule ID: {rule_id}")
print(f"Number of Active Rules: {num_active_rules}")
```

### Privacy Guarantees

The obfuscation mechanism nullifies statistics when:

1. **Case 1**: A fuzzy set has only 1-2 samples (risk of individual record inference)
2. **Case 2**: A fuzzy set is isolated (neighbors have zero samples)
3. **Case 3**: Root node statistics could expose overall distribution

Check obfuscation statistics:

```python
# In client logs
client.get_privacy_stats()
```

**Example Output:**
```
client_id: 0
total_obfuscation_checks: 450
obfuscation_applied_case1: 23  (5.1%)
obfuscation_applied_case2: 12  (2.7%)
obfuscation_applied_case3: 8   (1.8%)
```

---

## Troubleshooting

### Issue: Clients Cannot Connect to Director

**Symptoms:**
```
Error: Connection refused to director@172.16.2.185
```

**Solutions:**
1. Verify director is running: `docker logs director`
2. Check firewall rules allow ports 4369 and 9000
3. Verify IP addresses in environment variables match actual machine IPs
4. Ensure `FEDLANG_COOKIE` is identical across all nodes

### Issue: Dataset Not Found

**Symptoms:**
```
FileNotFoundError: /dataset/X_train.csv not found
```

**Solutions:**
1. Verify dataset partitions exist: `ls datasets_splits/WeatherIzimir/`
2. Check volume mounts in `docker-compose-clients.yml`:
   ```yaml
   volumes:
     - ./datasets_splits/WeatherIzimir/client_X_train_1.csv:/dataset/X_train.csv
   ```
3. Ensure paths are relative to docker-compose file location

### Issue: Federation Hangs During Execution

**Symptoms:**
- Federation stops at a specific round without error

**Solutions:**
1. Check client logs for errors: `docker logs client1`
2. Verify all clients are connected: count active clients in director logs
3. Increase network timeout: add `-kernel net_ticktime 120` to `ERL_FLAGS`

### Issue: Model File Not Saved

**Symptoms:**
```
ls /models/  # Empty directory
```

**Solutions:**
1. Check volume mount exists: `docker inspect director | grep models`
2. Verify permissions: `ls -ld /models` (should be writable)
3. Check server logs for write errors: `docker logs director`

---

## Next Steps

After completing this example, you can:

1. **Experiment with different algorithms**: Try Federated Fuzzy C-Means clustering
2. **Adjust hyperparameters**: Modify `num_fuzzy_sets`, `gain_threshold`, or `obfuscate`
3. **Use your own datasets**: Follow the data partitioning approach in test scripts
4. **Implement custom algorithms**: See the [CLAUDE.md](CLAUDE.md) guide for adding new algorithms
5. **Scale to more clients**: Add additional client services in docker-compose-clients.yml

---

## Additional Resources

- **Main README**: [README.md](README.md)
- **Development Guide**: [CLAUDE.md](CLAUDE.md)
- **GitHub Repository**: https://github.com/jlcorcuera/fedxai-lib
- **Research Papers**: See [Citations](README_DRAFT.md#citations) section

For questions or issues, please contact the [contributors](README_DRAFT.md#contributors) or open an issue on GitHub.
