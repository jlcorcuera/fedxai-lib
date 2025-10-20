#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
  @Filename:    load_plan_test
  @Author:      José Luis Corcuera Bárcena
  @Time:        7/4/25 12:08 PM
"""
import os
import pandas as pd
from fedxai_lib import run_fedxai_experiment, FedXAIAlgorithm
from fedxai_lib.algorithms.federated_frt.client import FedFRTClient
from fedxai_lib.algorithms.federated_frt.server import FedFRTServer
from fedxai_lib.algorithms.federated_frt.utils.robust_scaler import RobustScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

random_state = 19
num_clients = 5
iid_seed = 19
shuffle = True
LOW_PERCENTILE: float = 0.025
HIGH_PERCENTILE: float = 0.975

# (2) loading and dividing dataset into horizontal chunks
dataset_name = 'WeatherIzimir.csv'
ds_path = os.path.join('..', 'datasets', dataset_name)

split_folder = os.path.join('..', 'datasets_splits', dataset_name.split('.')[0])
os.makedirs(split_folder, exist_ok=True)

print(ds_path)
df = pd.read_csv(ds_path)

features = df.columns[:-1].values.tolist()
target = df.columns.values[-1]

# Normalize the data
scaler_x = RobustScaler(features, LOW_PERCENTILE, HIGH_PERCENTILE)
scaler_y = MinMaxScaler()

df_scaled = scaler_x.fit_transform(df)
df_scaled[target] = scaler_y.fit_transform(df_scaled[target].values.reshape(-1, 1)).ravel()

y = scaler_y.fit_transform(df_scaled[target].values.reshape(-1, 1)).ravel()
X = df_scaled.drop(columns=[target]).to_numpy()

dataset_by_client = {client_id: {
    'X_train': dict(),
    'y_train': dict(),
    'X_test': dict(),
    'y_test': dict()
} for client_id in range(0, num_clients)}

kf = KFold(n_splits=num_clients, random_state=random_state, shuffle=shuffle)
for fold_id, (train_index, test_index) in enumerate(kf.split(X)):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    kf_client = KFold(n_splits=num_clients, random_state=random_state, shuffle=True)

    for indexes, client_id in zip(kf_client.split(X_train), range(num_clients)):
        _, train_index_it = indexes

        df_split_X_train = pd.DataFrame(data=X_train[train_index_it], columns=features)
        df_split_y_train = pd.DataFrame(data=y_train[train_index_it], columns=[target])

        dataset_by_client[client_id]['X_train'] = df_split_X_train
        dataset_by_client[client_id]['y_train'] = df_split_y_train

        df_split_X_train.to_csv(os.path.join(split_folder, f'client_X_train_{client_id+1}.csv'), index=False)
        df_split_y_train.to_csv(os.path.join(split_folder, f'client_y_train_{client_id+1}.csv'), index=False)

    kf_client = KFold(n_splits=num_clients, random_state=random_state, shuffle=True)
    for indexes, client_id in zip(kf_client.split(X_test), range(num_clients)):
        _, test_index_it = indexes

        df_split_X_test = pd.DataFrame(data=X_test[test_index_it], columns=features)
        df_split_y_test = pd.DataFrame(data=y_test[test_index_it], columns=[target])

        dataset_by_client[client_id]['X_test'] = df_split_X_test
        dataset_by_client[client_id]['y_test'] = df_split_y_test

        df_split_X_test.to_csv(os.path.join(split_folder, f'client_X_test_{client_id + 1}.csv'), index=False)
        df_split_y_test.to_csv(os.path.join(split_folder, f'client_y_test_{client_id+1}.csv'), index=False)

    break

parameters = {
      "gain_threshold": 0.0001,
      "max_number_rounds": 100,
      "num_fuzzy_sets": 5,
      "max_depth": None,
      "min_samples_split_ratio": 0.1,
      "min_num_clients": 20,
      "obfuscate": True,
      "features_names": ["Max_temperature","Min_temperature","Dewpoint","Precipitation","Sea_level_pressure","Standard_pressure","Visibility","Wind_speed","Max_wind_speed"],
      "target": "Mean_temperature",
      "dataset_X_train": "/dataset/X_train.csv",
      "dataset_y_train": "/dataset/y_train.csv",
      "dataset_X_test": "/dataset/X_test.csv",
      "dataset_y_test": "/dataset/y_test.csv",
      "model_output_file": "/models/frt_weather_izimir.pickle"
}

clients = [FedFRTClient(type='client', id = idx, scaler_X=scaler_x, scaler_y=scaler_y,
                        X_train=dataset_by_client.get(idx)['X_train'],
                        y_train=dataset_by_client.get(idx)['y_train'],
                        X_test=dataset_by_client.get(idx)['X_test'],
                        y_test=dataset_by_client.get(idx)['y_test']) for idx in range(num_clients)]
server = FedFRTServer(type='server')

run_fedxai_experiment(FedXAIAlgorithm.FED_FRT_HORIZONTAL, server, clients, parameters)
