# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

import pickle
import numpy as np
import pandas as pd
from fedxai_lib import FedXAIAlgorithm, run_fedxai_experiment
from fedxai_lib.algorithms.federated_shap.server import FederatedShapServer
from fedxai_lib.algorithms.federated_shap.client import FederatedShapClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


def read_client_RMI_training_data(columns_desired, path_data):
    """Load RMI training data for a specific client."""
    df = pd.read_csv(path_data)
    X_train = df[columns_desired].to_numpy()
    y_train = df['Classe'].to_numpy()
    return X_train, y_train


dataset_name = 'RMI_demo'

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

num_clients = 3
num_features = 14

print(f"{'='*60}")
print(f"Loading RMI Dataset - {num_clients} clients")
print(f"{'='*60}")

path_train = "../datasets/RMI_demo/preprocessed/split_train_{id}.csv"

dataset_chunks = []
y_chunks = []

for client_id in range(num_clients):
    X_train, y_train = read_client_RMI_training_data(desired_columns, path_train.format(id=client_id))
    dataset_chunks.append(X_train)
    y_chunks.append(y_train)
    print(f"Client {client_id}: {X_train.shape[0]} samples, {X_train.shape[1]} features")

print(f"\nTotal samples: {sum(len(chunk) for chunk in dataset_chunks)}")
print(f"Features: {desired_columns}")


min_max_scaler = MinMaxScaler()
X_all = np.vstack(dataset_chunks)
min_max_scaler.fit(X_all)

dataset_chunks_normalized = [min_max_scaler.transform(chunk) for chunk in dataset_chunks]

parameters = {
    "num_clusters": 50,
    "centroid_seed": 0,
    "epsilon": 0.005,
    "lambda_factor": 2,
    "max_number_rounds": 40,
    "min_num_clients": 3,
    "dataset": "/datasets/RMI_demo/preprocessed/split_train_0.csv",
    "model_save_path": "../models/fedshap_rmi_model.pkl"
}

clients = [FederatedShapClient(type='client', id=idx, dataset=dataset_chunks_normalized[idx])
           for idx in range(num_clients)]
server = FederatedShapServer(type='server')

print(f"\n{'='*60}")
print("Running Federated SHAP Experiment")
print(f"{'='*60}")

run_fedxai_experiment(FedXAIAlgorithm.FED_SHAP, server, clients, parameters)

print(f"\n{'='*60}")
print("Federated SHAP Model Training Complete")
print(f"{'='*60}")


selected_client = clients[0]  # select first client for demonstration
shap_model = selected_client.get_model()
print(f"\nFedShapModel (from Client {selected_client.id}): {shap_model}")
print(f"Number of centroids (background dataset): {shap_model.num_clusters}")
print(f"Number of features: {shap_model.num_features}")
print(f"\nCentroids (first 3):")
for i, centroid in enumerate(shap_model.get_centroids()[:3]):
    print(f"  Cluster {i}: mean={np.mean(centroid):.4f}, std={np.std(centroid):.4f}")


print(f"\n{'='*60}")
print("Training a Neural Network for demonstration")
print(f"{'='*60}")


X_train = np.vstack(dataset_chunks_normalized)
y_train = np.hstack(y_chunks)

clf = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),  # 3 hidden layers
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    verbose=False
)
clf.fit(X_train, y_train)

train_accuracy = clf.score(X_train, y_train)
print(f"Neural Network architecture: Input({num_features}) -> 64 -> 32 -> 16 -> Output")
print(f"Training accuracy: {train_accuracy:.4f}")

X_test_path = "../datasets/RMI_demo/preprocessed/test.csv"
X_test_df = pd.read_csv(X_test_path)
y_test = X_test_df["Classe"].to_numpy()
X_test = X_test_df[desired_columns].to_numpy()
X_test_normalized = min_max_scaler.transform(X_test)

test_accuracy = clf.score(X_test_normalized, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

y_pred = clf.predict(X_test_normalized)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\n{'='*60}")
print("Computing SHAP values for test instances")
print(f"{'='*60}")

shap_model.set_predictor(clf)

np.random.seed(42)
test_indices = np.random.choice(len(X_test_normalized), size=min(5, len(X_test_normalized)), replace=False)
X_test_sample = X_test_normalized[test_indices]
y_test_sample = y_test[test_indices]

print("\nComputing SHAP values...")
shap_explanation = shap_model.explain(X_test_sample)

print("\nSHAP Values:")
print(f"Explanation type: {type(shap_explanation)}")
print(f"Base values shape: {np.array(shap_explanation.base_values).shape}")


shap_values = shap_explanation.values

print(f"SHAP values shape: {shap_values.shape}")

for i, (instance, true_label) in enumerate(zip(X_test_sample, y_test_sample)):
    pred_label = clf.predict(instance.reshape(1, -1))[0]
    pred_proba = clf.predict_proba(instance.reshape(1, -1))[0]

    print(f"\nInstance {i+1}:")
    print(f"  True label: {true_label}, Predicted: {pred_label}")
    print(f"  Prediction probabilities: {pred_proba}")

    # for multi-class
    if len(shap_values.shape) == 3:
        shap_vals_instance = shap_values[i, :, pred_label - 1]
        print(f"  SHAP values sum for predicted class {pred_label}: {np.sum(shap_vals_instance):.4f}")
        print(f"  Top 5 contributing features for predicted class:")
        top_features = np.argsort(np.abs(shap_vals_instance))[-5:][::-1]
        for feat_idx in top_features:
            print(f"    {desired_columns[feat_idx]}: {shap_vals_instance[feat_idx]:.4f}")
    else:
        shap_vals_instance = shap_values[i]
        print(f"  SHAP values sum: {np.sum(shap_vals_instance):.4f}")
        print(f"  Top 5 contributing features:")
        top_features = np.argsort(np.abs(shap_vals_instance))[-5:][::-1]
        for feat_idx in top_features:
            print(f"    {desired_columns[feat_idx]}: {shap_vals_instance[feat_idx]:.4f}")

print(f"\n{'='*60}")
print("Verifying model persistence")
print(f"{'='*60}")

model_path = parameters['model_save_path']
try:
    print(f"\nLoading saved model from: {model_path}")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    print(f"Loaded model: {loaded_model}")
    print(f"Centroids match: {np.allclose(loaded_model.get_centroids(), shap_model.get_centroids())}")


    loaded_model.set_predictor(clf)
    test_instance = X_test_sample[0].reshape(1, -1)
    shap_explanation_loaded = loaded_model.explain(test_instance)

    original_first_shap = shap_values[0]
    loaded_first_shap = shap_explanation_loaded.values[0]
    print(f"SHAP values from loaded model match: {np.allclose(loaded_first_shap, original_first_shap, atol=0.1)}")
except FileNotFoundError:
    print(f"Warning: Model file not found at {model_path}")

print(f"\n{'='*60}")
print("Test Complete!")
print(f"{'='*60}")