# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

import os
import pandas as pd
from fedxai_lib import FedXAIAlgorithm, run_fedxai_experiment
from fedxai_lib.algorithms.federated_cmeans_horizontal.client import FederatedHorizontalCMClient
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from fedxai_lib.algorithms.federated_cmeans_horizontal.server import FederatedHorizontalCMServer

num_clients = 20
iid_seed: int = 2

# (2) loading and dividing dataset into horizontal chunks
dataset_name = 'xclara.csv'
ds_path = os.path.join('..', 'datasets', dataset_name)

split_folder = os.path.join('..', 'datasets_splits', dataset_name.split('.')[0])
os.makedirs(split_folder, exist_ok=True)

print(ds_path)
df = pd.read_csv(ds_path)
features = df.columns.values[:-1].tolist()
target = df.columns[-1]
print(f'features: {features} \ntarget: {target}')
min_max_scaler = MinMaxScaler()
X_original = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
X = min_max_scaler.fit_transform(X_original)

dataset_chunks = []
y_chunks = []

skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=iid_seed)

for idx_split, (train_index, test_index) in enumerate(skf.split(X, Y), 1):
    dataset_chunks.append(X[test_index])
    y_chunks.append(Y[test_index])
    df_split = pd.DataFrame(data=X[test_index], columns=features)
    df_split.to_csv(os.path.join(split_folder, f'client_{idx_split}.csv'), index=False)

parameters = {
    "num_clusters": 3,
    "centroid_seed": 0,
    "epsilon": 0.005,
    "max_number_rounds": 100,
    "min_num_clients": 1,
    "dataset": "/datasets/xclara.csv"
}

clients = [FederatedHorizontalCMClient(type='client', id = idx, dataset=dataset_chunks[idx]) for idx in range(num_clients)]
server = FederatedHorizontalCMServer(type='server')

run_fedxai_experiment(FedXAIAlgorithm.FED_CMEANS_HORIZONTAL, server, clients, parameters)
