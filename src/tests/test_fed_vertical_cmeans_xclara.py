#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
  @Filename:    load_plan_test
  @Author:      José Luis Corcuera Bárcena
  @Time:        7/4/25 12:08 PM
"""
import os
import pandas as pd
from fedxai_lib.algorithms.federated_cmeans_vertical.client import FederatedVerticalCMClient
from fedxai_lib.algorithms.federated_cmeans_vertical.server import FederatedVerticalCMServer
from fedlangpy.core.utils import load_plan, run_experiment
from sklearn.preprocessing import MinMaxScaler

from fedxai_lib.descriptors.plan_loader import PlanEnum, load_fedxai_plan

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

parameters = {
    "num_clusters": 3,
    "centroid_seed": 0,
    "epsilon": 0.005,
    "lambda_factor": 2,
    "max_number_rounds": 100,
    "min_num_clients": 1,
    "dataset": "/datasets/xclara.csv"
}

fl_plan = load_fedxai_plan(PlanEnum.FED_CMEANS_VERTICAL)

clients = [FederatedVerticalCMClient(type='client', id = idx, dataset=dataset_chunks[idx]) for idx in range(num_clients)]
server = FederatedVerticalCMServer(type='server')

run_experiment(fl_plan, server, clients, parameters)
