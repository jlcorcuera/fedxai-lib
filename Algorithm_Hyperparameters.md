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

---

## Federated Fuzzy C-Means (Horizontal)

**Algorithm**: `FedXAIAlgorithm.FED_FCMEANS_HORIZONTAL`

**Local execution example**: [test_fed_fcmeans_horizontal_xclara.py](src/tests/test_fed_fcmeans_horizontal_xclara.py)

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

### Notes

- **Horizontal partitioning**: Each client holds a subset of data samples with all features.

---

## Federated C-Means (Horizontal)

**Algorithm**: `FedXAIAlgorithm.FED_CMEANS_HORIZONTAL`

**Local execution example**: [test_fed_cmeans_horizontal_xclara.py](src/tests/test_fed_cmeans_horizontal_xclara.py)

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

### Notes

- **Horizontal partitioning**: Each client holds a subset of data samples with all features.

---

## Federated Fuzzy C-Means (Vertical)

**Algorithm**: `FedXAIAlgorithm.FED_FCMEANS_VERTICAL`

**Local execution example**: [test_fed_fcmeans_vertical_xclara.py](src/tests/test_fed_fcmeans_vertical_xclara.py)

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

### Notes

- **Vertical partitioning**: Each client holds all data samples but only a subset of features.

---

## Federated C-Means (Vertical)

**Algorithm**: `FedXAIAlgorithm.FED_CMEANS_VERTICAL`

**Local execution example**: [test_fed_cmeans_vertical_xclara.py](src/tests/test_fed_cmeans_vertical_xclara.py)

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

### Notes

- **Vertical partitioning**: Each client holds all data samples but only a subset of features.

---

## Federated Fuzzy Regression Tree (FRT)

**Algorithm**: `FedXAIAlgorithm.FED_FRT_HORIZONTAL`

**Local execution example**: [test_fed_frt_weather_izimir.py](src/tests/test_fed_frt_weather_izimir.py)

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

### Notes

- **Horizontal partitioning**: Each client holds a subset of data samples with all features.

---

## Federated Rule-Based Classifier (FRBC)

**Algorithm**: `FedXAIAlgorithm.FED_FRBC_HORIZONTAL`

**Local execution example**: [test_frbc_RMI_DEMO_fedxai_lib.py](src/tests/test_frbc_RMI_DEMO_fedxai_lib.py)

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

### Notes

- **Horizontal partitioning**: Each client holds a subset of data samples with all features.

---

## Usage Example

```python
from fedxai_lib import run_fedxai_experiment, FedXAIAlgorithm
from fedxai_lib.algorithms.federated_frt.client import FedFRTClient
from fedxai_lib.algorithms.federated_frt.server import FedFRTServer

# Define hyperparameters from table above
parameters = {
    "gain_threshold": 0.0001,
    "max_number_rounds": 100,
    # ... other parameters
}

# Create clients and server
clients = [FedFRTClient(...) for idx in range(num_clients)]
server = FedFRTServer(type='server')

# Run experiment
run_fedxai_experiment(FedXAIAlgorithm.FED_FRT_HORIZONTAL, server, clients, parameters)
```

---

## References

For more details on algorithm implementations, see:

- **Main Documentation**: [README.md](README.md)
- **Illustrative Example**: [Illustrative_Example.md](Illustrative_Example.md)
- **Test Scripts**: [src/tests/](src/tests/)

For questions about hyperparameter selection or algorithm behavior, please refer to the research papers cited in the [README.md](README.md#citations) or contact the [contributors](README.md#contributors).
