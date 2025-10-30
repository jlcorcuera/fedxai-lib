# Algorithm Hyperparameters

This document provides detailed hyperparameter specifications for each algorithm implemented in fedxai-lib. These parameters can be configured when running federated experiments in both execution modes:

- **Local federation execution**: Parameters are passed to the `run_fedxai_experiment()` function as a Python dictionary
- **Docker-based distributed federation**: Parameters are specified in JSON execution configuration files (see `src/scripts/executions/` directory)

---

## Table of Contents

### Clustering Algorithms
1. [Federated Fuzzy C-Means (Horizontal)](#federated-fuzzy-c-means-horizontal)
2. [Federated C-Means (Horizontal)](#federated-c-means-horizontal)
3. [Federated Fuzzy C-Means (Vertical)](#federated-fuzzy-c-means-vertical)
4. [Federated C-Means (Vertical)](#federated-c-means-vertical)

### Tree-Based Algorithms
5. [Federated Fuzzy Regression Tree (FRT)](#federated-fuzzy-regression-tree-frt)

### Classification Algorithms
6. [Federated Rule-Based Classifier (FRBC)](#federated-rule-based-classifier-frbc)

### Explainability Algorithms
7. [Federated SHAP](#federated-shap)

---

## Federated Fuzzy C-Means (Horizontal)

**Algorithm**: `FedXAIAlgorithm.FED_FCMEANS_HORIZONTAL`

**Local execution example**: [test_fed_fcmeans_horizontal_xclara.py](src/tests/test_fed_fcmeans_horizontal_xclara.py)

### Description

Federated Fuzzy C-Means performs soft clustering where each data point can belong to multiple clusters with varying degrees of membership. Unlike hard clustering, fuzzy clustering assigns membership values between 0 and 1 to each cluster, making it suitable for datasets with overlapping clusters or uncertain boundaries. The algorithm operates in horizontal partitioning mode, where each client holds different samples with all features.

### Hyperparameters

| Parameter | Type | Default | Description | Example Value |
|-----------|------|---------|-------------|---------------|
| `num_clusters` | int | 3 | Number of clusters to discover in the data. | 3 |
| `centroid_seed` | int | 0 | Random seed for initial centroid initialization. Ensures reproducibility. | 0 |
| `epsilon` | float | 0.005 | Convergence threshold. Algorithm stops when centroid changes are below this value. | 0.005 |
| `lambda_factor` | float | 2 | Fuzziness parameter (also called fuzzifier or m). Controls degree of cluster fuzziness. Must be > 1. Higher values increase fuzziness. | 2 |
| `max_number_rounds` | int | 100 | Maximum number of iterations for the clustering algorithm. | 100 |
| `min_num_clients` | int | 1 | Minimum number of clients required to participate in each round. | 1 |
| `dataset` | str | required | Path to the dataset CSV file (used in Docker environment). | "/datasets/xclara.csv" |

### Example Configuration

```python
parameters = {
    "num_clusters": 3,
    "centroid_seed": 0,
    "epsilon": 0.005,
    "lambda_factor": 2,
    "max_number_rounds": 100,
    "min_num_clients": 1,
    "dataset": "/datasets/xclara.csv"
}
```

### Usage Pattern

```python
from fedxai_lib import FedXAIAlgorithm, run_fedxai_experiment
from fedxai_lib.algorithms.federated_fcmeans_horizontal.server import FederatedHorizontalFCMServer
from fedxai_lib.algorithms.federated_fcmeans_horizontal.client import FederatedHorizontalFCMClient
import numpy as np

# Prepare data partitions for each client
dataset_chunks = [...]  # List of numpy arrays, one per client

# Create clients and server
clients = [FederatedHorizontalFCMClient(type='client', id=idx, dataset=dataset_chunks[idx])
           for idx in range(num_clients)]
server = FederatedHorizontalFCMServer(type='server')

# Run federated fuzzy clustering
run_fedxai_experiment(FedXAIAlgorithm.FED_FCMEANS_HORIZONTAL, server, clients, parameters)

# Access results
final_centroids = server.cluster_centers[-1]  # Final cluster centroids
cluster_labels = server.get_cluster_labels()   # Cluster assignments
print(f"Discovered {len(final_centroids)} clusters")
```

---

## Federated C-Means (Horizontal)

**Algorithm**: `FedXAIAlgorithm.FED_CMEANS_HORIZONTAL`

**Local execution example**: [test_fed_cmeans_horizontal_xclara.py](src/tests/test_fed_cmeans_horizontal_xclara.py)

### Description

Federated C-Means performs hard clustering where each data point is assigned to exactly one cluster. This is the crisp (non-fuzzy) version of clustering, suitable for datasets with well-separated clusters and clear boundaries. The algorithm operates in horizontal partitioning mode, with clients holding different samples but all features.

### Hyperparameters

| Parameter | Type | Default | Description | Example Value |
|-----------|------|---------|-------------|---------------|
| `num_clusters` | int | 3 | Number of clusters to discover in the data. | 3 |
| `centroid_seed` | int | 0 | Random seed for initial centroid initialization. Ensures reproducibility. | 0 |
| `epsilon` | float | 0.005 | Convergence threshold. Algorithm stops when centroid changes are below this value. | 0.005 |
| `max_number_rounds` | int | 100 | Maximum number of iterations for the clustering algorithm. | 100 |
| `min_num_clients` | int | 1 | Minimum number of clients required to participate in each round. | 1 |
| `dataset` | str | required | Path to the dataset CSV file (used in Docker environment). | "/datasets/xclara.csv" |

### Example Configuration

```python
parameters = {
    "num_clusters": 3,
    "centroid_seed": 0,
    "epsilon": 0.005,
    "max_number_rounds": 100,
    "min_num_clients": 1,
    "dataset": "/datasets/xclara.csv"
}
```

### Usage Pattern

```python
from fedxai_lib import FedXAIAlgorithm, run_fedxai_experiment
from fedxai_lib.algorithms.federated_cmeans_horizontal.server import FederatedHorizontalCMServer
from fedxai_lib.algorithms.federated_cmeans_horizontal.client import FederatedHorizontalCMClient

# Prepare data partitions for each client
dataset_chunks = [...]  # List of numpy arrays, one per client

# Create clients and server
clients = [FederatedHorizontalCMClient(type='client', id=idx, dataset=dataset_chunks[idx])
           for idx in range(num_clients)]
server = FederatedHorizontalCMServer(type='server')

# Run federated hard clustering
run_fedxai_experiment(FedXAIAlgorithm.FED_CMEANS_HORIZONTAL, server, clients, parameters)

# Access results
final_centroids = server.cluster_centers[-1]
cluster_labels = server.get_cluster_labels()  # Hard assignments (each point to one cluster)
print(f"Discovered {len(final_centroids)} clusters")
```

---

## Federated Fuzzy C-Means (Vertical)

**Algorithm**: `FedXAIAlgorithm.FED_FCMEANS_VERTICAL`

**Local execution example**: [test_fed_fcmeans_vertical_xclara.py](src/tests/test_fed_fcmeans_vertical_xclara.py)

### Description

Federated Fuzzy C-Means (Vertical) performs soft clustering in scenarios where data is vertically partitioned across clients. In vertical partitioning, each client holds all samples but only a subset of features. This is common in scenarios where different organizations collect different types of measurements for the same set of subjects (e.g., different hospitals measuring different medical tests for the same patients). The algorithm computes fuzzy membership values while preserving data privacy by only sharing partial centroid computations.

### Hyperparameters

| Parameter | Type | Default | Description | Example Value |
|-----------|------|---------|-------------|---------------|
| `num_clusters` | int | 3 | Number of clusters to discover in the data. | 3 |
| `centroid_seed` | int | 0 | Random seed for initial centroid initialization. Ensures reproducibility. | 0 |
| `epsilon` | float | 0.005 | Convergence threshold. Algorithm stops when centroid changes are below this value. | 0.005 |
| `lambda_factor` | float | 2 | Fuzziness parameter (also called fuzzifier or m). Controls degree of cluster fuzziness. Must be > 1. Higher values increase fuzziness. | 2 |
| `max_number_rounds` | int | 100 | Maximum number of iterations for the clustering algorithm. | 100 |
| `min_num_clients` | int | 1 | Minimum number of clients required to participate in each round. | 1 |
| `dataset` | str | required | Path to the dataset CSV file (used in Docker environment). | "/datasets/xclara.csv" |

### Example Configuration

```python
parameters = {
    "num_clusters": 3,
    "centroid_seed": 0,
    "epsilon": 0.005,
    "lambda_factor": 2,
    "max_number_rounds": 100,
    "min_num_clients": 1,
    "dataset": "/datasets/xclara.csv"
}
```

### Usage Pattern

```python
from fedxai_lib import FedXAIAlgorithm, run_fedxai_experiment
from fedxai_lib.algorithms.federated_fcmeans_vertical.server import FederatedVerticalFCMServer
from fedxai_lib.algorithms.federated_fcmeans_vertical.client import FederatedVerticalFCMClient
import numpy as np

# Prepare feature partitions for each client (vertical split)
# Example: Client 0 has features 0-2, Client 1 has features 3-5, etc.
feature_splits = [
    data[:, 0:3],   # Client 0: features 0-2
    data[:, 3:6],   # Client 1: features 3-5
    data[:, 6:9]    # Client 2: features 6-8
]

# Create clients and server
clients = [FederatedVerticalFCMClient(type='client', id=idx, dataset=feature_splits[idx])
           for idx in range(num_clients)]
server = FederatedVerticalFCMServer(type='server')

# Run federated vertical fuzzy clustering
run_fedxai_experiment(FedXAIAlgorithm.FED_FCMEANS_VERTICAL, server, clients, parameters)

# Access results
final_centroids = server.cluster_centers[-1]  # Centroids with all features combined
print(f"Discovered {len(final_centroids)} clusters across {num_clients} feature partitions")
```

---

## Federated C-Means (Vertical)

**Algorithm**: `FedXAIAlgorithm.FED_CMEANS_VERTICAL`

**Local execution example**: [test_fed_cmeans_vertical_xclara.py](src/tests/test_fed_cmeans_vertical_xclara.py)

### Description

Federated C-Means (Vertical) performs hard clustering with vertical data partitioning, where each data point is assigned to exactly one cluster. Each client holds all samples but only a subset of features. This is useful when different organizations hold complementary feature sets for the same subjects (e.g., genomic data, clinical data, and imaging data for the same patients held by different institutions). The algorithm combines partial distance computations from each client to determine cluster assignments.

### Hyperparameters

| Parameter | Type | Default | Description | Example Value |
|-----------|------|---------|-------------|---------------|
| `num_clusters` | int | 3 | Number of clusters to discover in the data. | 3 |
| `centroid_seed` | int | 0 | Random seed for initial centroid initialization. Ensures reproducibility. | 0 |
| `epsilon` | float | 0.005 | Convergence threshold. Algorithm stops when centroid changes are below this value. | 0.005 |
| `lambda_factor` | float | 2 | Fuzziness parameter (included for API consistency, though this is a hard clustering algorithm). | 2 |
| `max_number_rounds` | int | 100 | Maximum number of iterations for the clustering algorithm. | 100 |
| `min_num_clients` | int | 1 | Minimum number of clients required to participate in each round. | 1 |
| `dataset` | str | required | Path to the dataset CSV file (used in Docker environment). | "/datasets/xclara.csv" |

### Example Configuration

```python
parameters = {
    "num_clusters": 3,
    "centroid_seed": 0,
    "epsilon": 0.005,
    "lambda_factor": 2,
    "max_number_rounds": 100,
    "min_num_clients": 1,
    "dataset": "/datasets/xclara.csv"
}
```

### Usage Pattern

```python
from fedxai_lib import FedXAIAlgorithm, run_fedxai_experiment
from fedxai_lib.algorithms.federated_cmeans_vertical.server import FederatedVerticalCMServer
from fedxai_lib.algorithms.federated_cmeans_vertical.client import FederatedVerticalCMClient
import numpy as np

# Prepare feature partitions for each client (vertical split)
# Example: Client 0 has features 0-2, Client 1 has features 3-5, etc.
feature_splits = [
    data[:, 0:3],   # Client 0: features 0-2
    data[:, 3:6],   # Client 1: features 3-5
    data[:, 6:9]    # Client 2: features 6-8
]

# Create clients and server
clients = [FederatedVerticalCMClient(type='client', id=idx, dataset=feature_splits[idx])
           for idx in range(num_clients)]
server = FederatedVerticalCMServer(type='server')

# Run federated vertical hard clustering
run_fedxai_experiment(FedXAIAlgorithm.FED_CMEANS_VERTICAL, server, clients, parameters)

# Access results
final_centroids = server.cluster_centers[-1]  # Centroids with all features combined
cluster_labels = server.get_cluster_labels()  # Hard assignments (each point to one cluster)
print(f"Discovered {len(final_centroids)} clusters across {num_clients} feature partitions")
```

---

## Federated Fuzzy Regression Tree (FRT)

**Algorithm**: `FedXAIAlgorithm.FED_FRT_HORIZONTAL`

**Local execution example**: [test_fed_frt_weather_izimir.py](src/tests/test_fed_frt_weather_izimir.py)

### Description

Federated Fuzzy Regression Tree builds interpretable regression models using fuzzy logic in a privacy-preserving federated setting. The algorithm grows a tree structure where each split uses fuzzy sets (linguistic terms like "low", "medium", "high") rather than crisp thresholds, making the model more interpretable and robust to noise. Each leaf node contains a linear regression model. The algorithm includes privacy-preserving obfuscation mechanisms that prevent individual data point reconstruction while allowing collaborative model building across distributed clients.

### Hyperparameters

| Parameter | Type | Default | Description | Example Value |
|-----------|------|---------|-------------|---------------|
| `gain_threshold` | float | 0.0001 | Minimum information gain required for a node split. Splits with gain below this threshold are not performed. | 0.0001 |
| `max_number_rounds` | int | 100 | Maximum number of tree growing rounds (iterations). Controls the tree depth indirectly. | 100 |
| `num_fuzzy_sets` | int | 5 | Number of fuzzy sets to partition each feature domain. Higher values create more granular fuzzy partitions. | 5 |
| `max_depth` | int or None | None | Maximum depth of the tree. If `None`, tree grows until stopping criteria are met. | None |
| `min_samples_split_ratio` | float | 0.1 | Minimum ratio of samples required to split a node (relative to total samples). Prevents splits on small sample sets. | 0.1 |
| `min_num_clients` | int | 20 | Minimum number of samples required for privacy-preserving obfuscation. Statistics with fewer samples are nullified. | 20 |
| `obfuscate` | bool | True | Enable privacy obfuscation mechanisms. When True, nullifies statistics that could reveal individual records. | True |
| `features_names` | list[str] | required | List of input feature names in the dataset. Used for model interpretation. | ["Max_temperature", "Min_temperature", ...] |
| `target` | str | required | Name of the target variable to predict. | "Mean_temperature" |
| `dataset_X_train` | str | required | Path to training features CSV file (client-side path in Docker). | "/dataset/X_train.csv" |
| `dataset_y_train` | str | required | Path to training target CSV file (client-side path in Docker). | "/dataset/y_train.csv" |
| `dataset_X_test` | str | required | Path to test features CSV file (client-side path in Docker). | "/dataset/X_test.csv" |
| `dataset_y_test` | str | required | Path to test target CSV file (client-side path in Docker). | "/dataset/y_test.csv" |
| `model_output_file` | str | required | Path where the trained model will be saved (server-side path). | "/models/frt_weather_izimir.pickle" |

### Example Configuration

```python
parameters = {
    "gain_threshold": 0.0001,
    "max_number_rounds": 100,
    "num_fuzzy_sets": 5,
    "max_depth": None,
    "min_samples_split_ratio": 0.1,
    "min_num_clients": 20,
    "obfuscate": True,
    "features_names": [
        "Max_temperature", "Min_temperature", "Dewpoint",
        "Precipitation", "Sea_level_pressure", "Standard_pressure",
        "Visibility", "Wind_speed", "Max_wind_speed"
    ],
    "target": "Mean_temperature",
    "dataset_X_train": "/dataset/X_train.csv",
    "dataset_y_train": "/dataset/y_train.csv",
    "dataset_X_test": "/dataset/X_test.csv",
    "dataset_y_test": "/dataset/y_test.csv",
    "model_output_file": "/models/frt_weather_izimir.pickle"
}
```

### Usage Pattern

```python
from fedxai_lib import FedXAIAlgorithm, run_fedxai_experiment
from fedxai_lib.algorithms.federated_frt.server import FedFRTServer
from fedxai_lib.algorithms.federated_frt.client import FedFRTClient
from sklearn.preprocessing import StandardScaler
import pickle

# Prepare data partitions for each client
# Each client has their own X_train, y_train, X_test, y_test
dataset_by_client = {...}  # Dictionary mapping client_id to data splits

# Create scalers (shared across all clients for consistency)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit scalers on combined training data
scaler_X.fit(X_train_combined)
scaler_y.fit(y_train_combined.reshape(-1, 1))

# Create clients with their local data
clients = [
    FedFRTClient(
        type='client',
        id=idx,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        X_train=dataset_by_client[idx]['X_train'],
        y_train=dataset_by_client[idx]['y_train'],
        X_test=dataset_by_client[idx]['X_test'],
        y_test=dataset_by_client[idx]['y_test']
    )
    for idx in range(num_clients)
]

# Create server
server = FedFRTServer(type='server')

# Run federated tree growing
run_fedxai_experiment(FedXAIAlgorithm.FED_FRT_HORIZONTAL, server, clients, parameters)

# Load and use the trained model
with open(parameters['model_output_file'], 'rb') as f:
    frt_model = pickle.load(f)

# Make predictions
predictions = frt_model.predict(X_new)

# Print tree structure for interpretation
frt_model.print_tree()
```

---

## Federated Rule-Based Classifier (FRBC)

**Algorithm**: `FedXAIAlgorithm.FED_FRBC_HORIZONTAL`

**Local execution example**: [test_fed_rbc_rmi_demo_fedxai_lib.py](src/tests/test_fed_rbc_rmi_demo_fedxai_lib.py)

### Description

Federated Rule-Based Classifier builds interpretable classification models using fuzzy rules in a privacy-preserving federated setting. The algorithm generates a rule base where each rule has linguistic terms as antecedents (e.g., "IF feature1 is LOW AND feature2 is HIGH THEN class = 1"). Rules are extracted from distributed data and aggregated federally based on their confidence and support. The resulting model is highly interpretable, allowing domain experts to understand and validate the decision-making process. Privacy obfuscation mechanisms prevent individual data exposure while enabling collaborative rule discovery.

### Hyperparameters

| Parameter | Type | Default | Description | Example Value |
|-----------|------|---------|-------------|---------------|
| `num_fuzzy_sets` | int | 5 | Number of fuzzy sets to partition each feature domain. Defines the granularity of linguistic terms for rule antecedents. | 5 |
| `num_features` | int | required | Number of input features in the dataset. Must match the dimensionality of the input data. | 14 |
| `max_number_rounds` | int | 1 | Number of training rounds for federated rule base construction. Typically 1 for FRBC. | 1 |
| `min_num_clients` | int | 3 | Minimum number of clients required to participate in the federation. | 3 |
| `obfuscate` | bool | True | Enable privacy obfuscation mechanisms to protect individual client data contributions. | True |
| `target` | str | required | Name of the target class variable to predict. | "class" |
| `desired_columns` | list[str] | required | List of feature column names to use from the dataset. Specifies which features participate in rule construction. | ["feature1", "feature2", ...] |
| `dataset_X_train` | str | required | Path to training dataset CSV file (client-side path in Docker). Should contain both features and target class. | "/dataset/X_train.csv" |
| `output_model_folder` | str | "/tmp" | Directory path for temporary model outputs during training. | "/tmp" |
| `model_output_file` | str | required | Path where the trained FRBC model will be saved (server-side path). | "/models/frbc_model.pickle" |

### Example Configuration

```python
parameters = {
    "max_number_rounds": 1,
    "num_fuzzy_sets": 5,
    "min_num_clients": 3,
    "num_features": 14,
    "obfuscate": True,
    "target": "class",
    "dataset_X_train": "/dataset/X_train.csv",
    "output_model_folder": "/tmp",
    "model_output_file": "/models/frbc_RMI.pickle",
    "desired_columns": [
        "original_shape2D_Elongation",
        "original_shape2D_MajorAxisLength",
        "original_shape2D_MinorAxisLength",
        "original_shape2D_Perimeter",
        "original_shape2D_MaximumDiameter",
        "original_shape2D_Sphericity",
        "original_firstorder_Mean",
        "original_firstorder_Median",
        "original_firstorder_Minimum",
        "original_firstorder_Maximum",
        "original_firstorder_Range",
        "original_firstorder_Uniformity",
        "Centroid_X",
        "Centroid_Y"
    ]
}
```

### Usage Pattern

```python
from fedxai_lib import FedXAIAlgorithm, run_fedxai_experiment
from fedxai_lib.algorithms.federated_frbc.server import FedFRBCServer
from fedxai_lib.algorithms.federated_frbc.client import FedFRBCClient
import pandas as pd
import pickle

# Prepare data partitions for each client
# Each client has their training data with features + target class
client_data_paths = [...]  # List of paths to client CSV files

# Create clients with their local data
clients = [
    FedFRBCClient(
        type='client',
        id=idx,
        dataset=pd.read_csv(path)
    )
    for idx, path in enumerate(client_data_paths)
]

# Create server
server = FedFRBCServer(type='server')

# Run federated rule extraction
run_fedxai_experiment(FedXAIAlgorithm.FED_FRBC_HORIZONTAL, server, clients, parameters)

# Load and use the trained model
with open(parameters['model_output_file'], 'rb') as f:
    frbc_model = pickle.load(f)

# Make predictions
predictions = frbc_model.predict(X_new)

# Print rules for interpretation
for idx, rule in enumerate(frbc_model.rules):
    print(f"Rule {idx+1}: {rule}")
```

---

## Federated SHAP

**Algorithm**: `FedXAIAlgorithm.FED_SHAP`

**Local execution example**: [test_fed_shap_rmi.py](src/tests/test_fed_shap_rmi.py)

### Description

Federated SHAP provides consistent post-hoc explainability for any machine learning model trained in federated settings. It builds a privacy-preserving background dataset using Federated Fuzzy C-means clustering, then uses the SHAP library to compute model-agnostic feature attributions. This approach ensures explanation consistency across clients without sharing raw data.

### Hyperparameters

This algorithm extends [Federated Fuzzy C-Means (Horizontal)](#federated-fuzzy-c-means-horizontal) and inherits its parameters:

| Parameter | Type | Default | Description | Example Value |
|-----------|------|---------|-------------|---------------|
| `num_clusters` | int | 5 | Number of clusters (centroids) to use as background dataset for SHAP. More clusters provide finer-grained background distribution. | 5 |
| `centroid_seed` | int | 0 | Random seed for initial centroid initialization. Ensures reproducibility. | 0 |
| `epsilon` | float | 0.005 | Convergence threshold for clustering. Algorithm stops when centroid changes are below this value. | 0.005 |
| `lambda_factor` | float | 2 | Fuzziness parameter for clustering. Must be > 1. Higher values increase fuzziness. | 2 |
| `max_number_rounds` | int | 100 | Maximum number of clustering iterations. | 100 |
| `min_num_clients` | int | 3 | Minimum number of clients required to participate in each round. | 3 |
| `dataset` | str | required | Path to the dataset CSV file (used in Docker environment). | "/datasets/RMI_demo/preprocessed/split_train_0.csv" |
| `model_save_path` | str | optional | Path where the FedShapModel will be saved. | "../models/fedshap_rmi_model.pkl" |

### Example Configuration

```python
parameters = {
    "num_clusters": 5,
    "centroid_seed": 0,
    "epsilon": 0.005,
    "lambda_factor": 2,
    "max_number_rounds": 100,
    "min_num_clients": 3,
    "dataset": "/datasets/RMI_demo/preprocessed/split_train_0.csv",
    "model_save_path": "../models/fedshap_rmi_model.pkl"
}
```

### Usage Pattern

```python
from fedxai_lib import FedXAIAlgorithm, run_fedxai_experiment
from fedxai_lib.algorithms.federated_shap.server import FederatedShapServer
from fedxai_lib.algorithms.federated_shap.client import FederatedShapClient
from sklearn.neural_network import MLPClassifier

# Create clients and server
clients = [FederatedShapClient(type='client', id=idx, dataset=client_data[idx])
           for idx in range(num_clients)]
server = FederatedShapServer(type='server')

# Run federated clustering to build background dataset
run_fedxai_experiment(FedXAIAlgorithm.FED_SHAP, server, clients, parameters)

# Get the FedShapModel from a client
shap_model = clients[0].get_model()

# Train your model (e.g., Neural Network)
model = MLPClassifier(hidden_layer_sizes=(64, 32, 16))
model.fit(X_train, y_train)

# Set the model to explain and compute SHAP values
shap_model.set_predictor(model)
shap_explanation = shap_model.explain(X_test)

# Access SHAP values
shap_values = shap_explanation.values  # Shape: (n_instances, n_features) or (n_instances, n_features, n_classes)
base_values = shap_explanation.base_values  # Expected value(s)
```

---

## References

For more details on algorithm implementations, see:

- **Main Documentation**: [README.md](README.md)
- **Illustrative Example**: [Illustrative_Example.md](Illustrative_Example.md)
- **Test Scripts**: [src/tests/](src/tests/)

For questions about hyperparameter selection or algorithm behavior, please refer to the research papers cited in the [README.md](README.md#citations) or contact the [contributors](README.md#contributors).
