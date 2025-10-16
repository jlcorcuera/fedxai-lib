#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
  @Filename:    load_plan_test
  @Author:      José Luis Corcuera Bárcena
  @Time:        7/4/25 12:08 PM
"""
import os
import pandas as pd
from fedxai_lib.algorithms.federated_fcmeans_horizontal.server import FederatedHorizontalFCMServer
from fedxai_lib.algorithms.federated_fcmeans_horizontal.client import FederatedHorizontalFCMClient
from fedlangpy.core.utils import load_plan, run_experiment
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler



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
    df_split.to_csv(os.path.join(split_folder, f'client_horizontal_{idx_split}.csv'), index=False)


plan_path = './plans/plan_federated_fcmeans_horizontal_xclara.json'

fl_plan = load_plan(plan_path, server_class=FederatedHorizontalFCMServer, client_class=FederatedHorizontalFCMClient)

clients = [FederatedHorizontalFCMClient(type='client', id = idx, dataset=dataset_chunks[idx]) for idx in range(num_clients)]
server = FederatedHorizontalFCMServer(type='server')

run_experiment(fl_plan, server, clients)