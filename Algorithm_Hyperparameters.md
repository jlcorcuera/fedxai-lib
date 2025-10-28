# Algorithm Hyperparameters

This document provides detailed hyperparameter specifications for each algorithm implemented in fedxai-lib. These parameters can be configured when running federated experiments using the `run_fedxai_experiment()` function.

---

## Table of Contents

1. [Federated Fuzzy Regression Tree (FRT)](#federated-fuzzy-regression-tree-frt)
2. [Federated Fuzzy C-Means (Horizontal)](#federated-fuzzy-c-means-horizontal)
3. [Federated Fuzzy C-Means (Vertical)](#federated-fuzzy-c-means-vertical)
4. [Federated C-Means (Horizontal)](#federated-c-means-horizontal)
5. [Federated C-Means (Vertical)](#federated-c-means-vertical)

---

## Federated Fuzzy Regression Tree (FRT)

**Algorithm**: `FedXAIAlgorithm.FED_FRT_HORIZONTAL`

**Test File**: [test_fed_frt_weather_izimir.py](src/tests/test_fed_frt_weather_izimir.py)

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

---

## Federated Fuzzy C-Means (Horizontal)

**Algorithm**: `FedXAIAlgorithm.FED_FCMEANS_HORIZONTAL`

**Test File**: [test_fed_fcmeans_horizontal_xclara.py](src/tests/test_fed_fcmeans_horizontal_xclara.py)

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
- The `lambda_factor` (fuzzifier) controls the degree of fuzzy overlap between clusters. A value of 2 is commonly used as a balanced choice.

---

## Federated Fuzzy C-Means (Vertical)

**Algorithm**: `FedXAIAlgorithm.FED_FCMEANS_VERTICAL`

**Test File**: [test_fed_fcmeans_vertical_xclara.py](src/tests/test_fed_fcmeans_vertical_xclara.py)

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
- Distance computations are performed in a privacy-preserving manner without revealing feature values.

---

## Federated C-Means (Horizontal)

**Algorithm**: `FedXAIAlgorithm.FED_CMEANS_HORIZONTAL`

**Test File**: [test_fed_cmeans_horizontal_xclara.py](src/tests/test_fed_cmeans_horizontal_xclara.py)

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
- Unlike Fuzzy C-Means, this is a hard clustering algorithm where each sample belongs to exactly one cluster.
- No `lambda_factor` parameter since this is not a fuzzy algorithm.

---

## Federated C-Means (Vertical)

**Algorithm**: `FedXAIAlgorithm.FED_CMEANS_VERTICAL`

**Test File**: [test_fed_cmeans_vertical_xclara.py](src/tests/test_fed_cmeans_vertical_xclara.py)

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
- This is a hard clustering algorithm where each sample belongs to exactly one cluster.
- Distance computations are performed in a privacy-preserving manner without revealing feature values.

---

## General Notes

### Parameter Tuning Guidelines

1. **Clustering Algorithms** (`num_clusters`, `epsilon`):
   - Start with `num_clusters` based on domain knowledge or elbow method analysis
   - Adjust `epsilon` for convergence speed vs. accuracy trade-off
   - Smaller `epsilon` values lead to more precise but slower convergence

2. **Fuzzy Algorithms** (`lambda_factor`):
   - Common values range from 1.5 to 3
   - Higher values increase overlap between clusters
   - Value of 2 is most commonly used in literature

3. **Tree Algorithms** (`gain_threshold`, `num_fuzzy_sets`):
   - Lower `gain_threshold` creates deeper, more complex trees
   - More `num_fuzzy_sets` increases interpretability but computational cost
   - Balance `obfuscate` privacy vs. model accuracy based on privacy requirements

### Privacy Considerations

- **FRT Obfuscation**: When `obfuscate=True`, statistics from very small sample sets are nullified to prevent inference attacks
- **min_num_clients**: Higher values provide stronger privacy guarantees but may limit model expressiveness
- All algorithms ensure raw data never leaves client nodes

### Performance Considerations

- **max_number_rounds**: Set based on dataset size and complexity
- **Convergence**: Monitor logs to check if algorithms converge before max_number_rounds
- **Docker paths**: Dataset and model paths in parameters should match Docker volume mounts

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
- **Development Guide**: [CLAUDE.md](CLAUDE.md)
- **Illustrative Example**: [Illustrative_Example.md](Illustrative_Example.md)
- **Test Scripts**: [src/tests/](src/tests/)

For questions about hyperparameter selection or algorithm behavior, please refer to the research papers cited in the [README.md](README.md#citations) or contact the [contributors](README.md#contributors).
