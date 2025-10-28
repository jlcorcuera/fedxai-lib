import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from fedxai_lib import FedXAIAlgorithm, run_fedxai_experiment
from fedxai_lib.algorithms.federated_frbc.client import FederatedFRBCClient
from fedxai_lib.algorithms.federated_frbc.frbcs_fedxai_lib import FRBC_no_opt
from fedxai_lib.algorithms.federated_frbc.server import FederatedFRBCServer


def read_client_RMI_training_data(columns_desired, path_data):
    # path_magic = './datasets/RMI_demo/X_train.csv'

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


parameters = {
    "max_number_rounds": 1,
    "num_fuzzy_sets": 5,
    "min_num_clients": 3,
    "num_features": 14,
    "obfuscate": True,
    "target": "class",
    "output_model_folder": "/tmp",
    "model_output_file": "../models/frbc_RMI.pickle",
    "desired_columns": desired_columns}

num_clients = 3
num_features = 14
num_fuzzy_sets = 5


path_train = "../datasets/RMI_demo/preprocessed/split_train_{id}.csv"

clients = []
for client in range(num_clients):

    X_train, y_train = read_client_RMI_training_data(desired_columns, path_train.format(id=client))
    
    clients.append(FederatedFRBCClient(type='client', id=client,
                                        X_train = X_train, y_train = y_train))


server = FederatedFRBCServer(type='server')

run_fedxai_experiment(FedXAIAlgorithm.FED_FRBC_HORIZONTAL, server, clients, parameters)


print("Experiment completed successfully.")


with open("../models/frbc_RMI.pickle", 'rb') as f:
    frbc_model = pickle.load(f)

# ## CENTRALIZED

X_test_path = "../datasets/RMI_demo/preprocessed/test.csv"
X_test = pd.read_csv(X_test_path)

y_test = X_test["Classe"]
X_test = X_test[desired_columns]

y_pred_train = frbc_model.predict_and_get_rule_parallelized(X_test.values, "CF")
y_pred = np.array(y_pred_train, dtype=object)[:,0]
y_pred_clean = pd.Series(y_pred).astype(int).to_numpy()


print(classification_report(y_test.values, y_pred_clean, output_dict=True))


# with open("./save_rules.txt", "w") as f:
#     for predicted_class, rule_idx in y_pred_train:
#         rule = frbc_model.get_rule(rule_idx)
#         f.write(f"y: {predicted_class}, rule: {rule}\n")
