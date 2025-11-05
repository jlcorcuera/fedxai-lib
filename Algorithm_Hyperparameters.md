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

**Identifier**: federated_fcmeans_horizontal

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

### Code example

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

**Identifier**: federated_cmeans_horizontal

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

### Code example

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

**Identifier**: federated_fcmeans_vertical

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

### Code example

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

**Identifier**: federated_cmeans_vertical

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

### Code example

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

**Identifier**: federated_frt

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

### Code example

```python
import os
import pickle
import pandas as pd
from fedxai_lib import run_fedxai_experiment, FedXAIAlgorithm
from fedxai_lib.algorithms.federated_frt.client import FedFRTClient
from fedxai_lib.algorithms.federated_frt.server import FedFRTServer
from fedxai_lib.algorithms.federated_frt.utils.robust_scaler import RobustScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# Load dataset
dataset_name = 'WeatherIzimir.csv'
ds_path = os.path.join('..', 'datasets', dataset_name)
df = pd.read_csv(ds_path)

features = df.columns[:-1].values.tolist()
target = df.columns.values[-1]

# Create scalers for normalization
LOW_PERCENTILE = 0.025
HIGH_PERCENTILE = 0.975
scaler_x = RobustScaler(features, LOW_PERCENTILE, HIGH_PERCENTILE)
scaler_y = MinMaxScaler()

# Normalize data
df_scaled = scaler_x.fit_transform(df)
df_scaled[target] = scaler_y.fit_transform(df_scaled[target].values.reshape(-1, 1)).ravel()

y = df_scaled[target].values.reshape(-1, 1).ravel()
X = df_scaled.drop(columns=[target]).to_numpy()

# Split data across clients using KFold
num_clients = 5
random_state = 19
dataset_by_client = {client_id: {
    'X_train': dict(), 'y_train': dict(),
    'X_test': dict(), 'y_test': dict()
} for client_id in range(num_clients)}

kf = KFold(n_splits=num_clients, random_state=random_state, shuffle=True)
for fold_id, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    kf_client = KFold(n_splits=num_clients, random_state=random_state, shuffle=True)
    for indexes, client_id in zip(kf_client.split(X_train), range(num_clients)):
        _, train_index_it = indexes
        dataset_by_client[client_id]['X_train'] = pd.DataFrame(X_train[train_index_it], columns=features)
        dataset_by_client[client_id]['y_train'] = pd.DataFrame(y_train[train_index_it], columns=[target])

    kf_client = KFold(n_splits=num_clients, random_state=random_state, shuffle=True)
    for indexes, client_id in zip(kf_client.split(X_test), range(num_clients)):
        _, test_index_it = indexes
        dataset_by_client[client_id]['X_test'] = pd.DataFrame(X_test[test_index_it], columns=features)
        dataset_by_client[client_id]['y_test'] = pd.DataFrame(y_test[test_index_it], columns=[target])
    break

# Define parameters
parameters = {
    "gain_threshold": 0.0001,
    "max_number_rounds": 100,
    "num_fuzzy_sets": 5,
    "max_depth": None,
    "min_samples_split_ratio": 0.1,
    "min_num_clients": 20,
    "obfuscate": True,
    "features_names": features,
    "target": target,
    "dataset_X_train": "/dataset/X_train.csv",
    "dataset_y_train": "/dataset/y_train.csv",
    "dataset_X_test": "/dataset/X_test.csv",
    "dataset_y_test": "/dataset/y_test.csv",
    "model_output_file": "/models/frt_weather_izimir.pickle"
}

# Create clients and server
clients = [FedFRTClient(type='client', id=idx, scaler_X=scaler_x, scaler_y=scaler_y,
                        X_train=dataset_by_client[idx]['X_train'],
                        y_train=dataset_by_client[idx]['y_train'],
                        X_test=dataset_by_client[idx]['X_test'],
                        y_test=dataset_by_client[idx]['y_test'])
           for idx in range(num_clients)]
server = FedFRTServer(type='server')

# Run federated tree growing
run_fedxai_experiment(FedXAIAlgorithm.FED_FRT_HORIZONTAL, server, clients, parameters)

# Load and use the trained model
with open(parameters['model_output_file'], 'rb') as f:
    model = pickle.load(f)

# Make predictions and access activated rules
X_test_client = dataset_by_client[0]['X_test']
y_test_client = scaler_y.inverse_transform(dataset_by_client[0]['y_test'].values.reshape(-1, 1)).ravel()

y_predict_with_rules = model.predict(X_test_client.values)
y_predict = scaler_y.inverse_transform(y_predict_with_rules[:, 0].reshape(-1, 1)).ravel()
activated_rules = y_predict_with_rules[:, 1].astype(int)

# Print predictions with activated rules
for y_true, y_pred, rule_idx in zip(y_test_client, y_predict, activated_rules):
    print(f"True: {y_true}, Predicted: {y_pred}, Rule: {model.get_rule_by_index(rule_idx)}")
```

---

## Federated Rule-Based Classifier (FRBC)

**Identifier**: federated_frbc

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

### Code example

```python
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from fedxai_lib import FedXAIAlgorithm, run_fedxai_experiment
from fedxai_lib.algorithms.federated_frbc.client import FederatedFRBCClient
from fedxai_lib.algorithms.federated_frbc.server import FederatedFRBCServer

# Define feature columns to use
desired_columns = [
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

# Define parameters
parameters = {
    "max_number_rounds": 1,
    "num_fuzzy_sets": 5,
    "min_num_clients": 3,
    "num_features": 14,
    "obfuscate": True,
    "target": "class",
    "output_model_folder": "/tmp",
    "model_output_file": "../models/frbc_RMI.pickle",
    "desired_columns": desired_columns,
    "unique_labels": {
        "1": "Meningioma",
        "2": "Glioma",
        "3": "Pituitary"
    },
    "feature_names": desired_columns
}

# Load and prepare client data
num_clients = 3
path_train = "../datasets/RMI_demo/preprocessed/split_train_{id}.csv"

clients = []
for client_id in range(num_clients):
    # Read training data for each client
    df = pd.read_csv(path_train.format(id=client_id))
    X_train = df[desired_columns].to_numpy()
    y_train = df['Classe'].to_numpy()

    # Create client entity
    clients.append(FederatedFRBCClient(type='client', id=client_id,
                                       X_train=X_train, y_train=y_train))

# Create server
server = FederatedFRBCServer(type='server')

# Run federated rule extraction
run_fedxai_experiment(FedXAIAlgorithm.FED_FRBC_HORIZONTAL, server, clients, parameters)

print("Experiment completed successfully.")

# Load and use the trained model
with open("../models/frbc_RMI.pickle", 'rb') as f:
    frbc_model = pickle.load(f)

# Test the model
X_test_path = "../datasets/RMI_demo/preprocessed/test.csv"
X_test_df = pd.read_csv(X_test_path)

y_test = X_test_df["Classe"]
X_test = X_test_df[desired_columns]

# Make predictions with activated rules
y_pred_train = frbc_model.predict(X_test.values)
y_pred = np.array(y_pred_train, dtype=object)[:, 0]
y_pred_clean = pd.Series(y_pred).astype(int).to_numpy()

# Evaluate model
print(classification_report(y_test.values, y_pred_clean, output_dict=True))

# Print activated rules for each prediction
with open("./frbc_rmi_demo_test_rules_dump.txt", "w") as f:
    for predicted_class, rule_idx in y_pred_train:
        rule_text = frbc_model.get_rule_by_index(rule_idx)
        f.write(f"{rule_text}\n")
        print(f"Predicted class: {predicted_class}, Rule: {rule_text}")
```

---

## Federated SHAP

**Identifier**: federated_shap

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

### Code example

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
