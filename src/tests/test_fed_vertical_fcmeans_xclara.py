#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
  @Filename:    load_plan_test
  @Author:      José Luis Corcuera Bárcena
  @Time:        7/4/25 12:08 PM
"""
import os
import numpy as np
import pandas as pd
from fedxai_lib.algorithms.federated_fcmeans_vertical.client import FederatedVerticalFCMClient
from fedxai_lib.algorithms.federated_fcmeans_vertical.server import FederatedVerticalFCMServer
from fedlangpy.core.utils import load_plan, run_experiment
from sklearn.preprocessing import MinMaxScaler

num_clients = 2
features_per_client = [1, 1]
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
last_index = 0
for i in range(num_clients):
    cstart_index = last_index
    clast_index = cstart_index + features_per_client[i]
    dataset_chunks.append(X[:, cstart_index:clast_index])
    last_index = clast_index


plan_path = './plans/plan_federated_fcmeans_vertical_xclara.json'

fl_plan = load_plan(plan_path, server_class=FederatedVerticalFCMServer, client_class=FederatedVerticalFCMClient)

clients = [FederatedVerticalFCMClient(type='client', id = idx, dataset=dataset_chunks[idx]) for idx in range(num_clients)]
server = FederatedVerticalFCMServer(type='server')

run_experiment(fl_plan, server, clients)
