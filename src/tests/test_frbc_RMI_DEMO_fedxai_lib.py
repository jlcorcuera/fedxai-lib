import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from fedxai_lib import FedXAIAlgorithm, run_fedxai_experiment
from fedxai_lib.algorithms.federated_frbc.client import FederatedFRBCClient
from fedxai_lib.algorithms.federated_frbc.frbcs_dynamic_fs_gpu import \
    FRBC_no_opt
from fedxai_lib.algorithms.federated_frbc.server import FederatedFRBCServer


def read_client_RMI_training_data(columns_desired):
    path_magic = './datasets/RMI_demo/X_train.csv'

    df = pd.read_csv(path_magic)
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


parameters = {
    "max_number_rounds": 1,
    "num_fuzzy_sets": 5,
    "min_num_clients": 3,
    "num_features": 14,
    "obfuscate": True,
    "target": "class",
    "output_model_folder": "/tmp",
    "model_output_file": "./models/frbc_RMI.pickle",
    "desired_columns": desired_columns}

num_clients = 3
num_features = 14
num_fuzzy_sets = 5



path_train = './datasets/RMI_demo/X_train.csv'

clients = []
for client in range(num_clients):
    X_train, y_train = read_client_RMI_training_data(desired_columns)
    # df_X_train = pd.DataFrame(data=X_train)
    # df_y_train = pd.DataFrame(data=y_train, columns=['class'])

    clients.append(FederatedFRBCClient(type='client', id=client,
                                        dataset_X_train=path_train))


server = FederatedFRBCServer(type='server')

run_fedxai_experiment(FedXAIAlgorithm.FED_FRBC_HORIZONTAL, server, clients, parameters)


