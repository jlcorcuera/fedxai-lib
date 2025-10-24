import numpy as np
import pandas as pd
from fedlangpy.algorithms.federated_frbc.client import FederatedFRBCClient
from fedlangpy.algorithms.federated_frbc.frbcs_dynamic_fs_gpu import \
    FRBC_no_opt
from fedlangpy.algorithms.federated_frbc.server import FederatedFRBCServer
from fedlangpy.core.utils import load_plan, run_experiment
from sklearn.metrics import classification_report


def read_client_RMI_training_data(client_id, dataset, columns_desired):
    path_magic = f'./datasets/{dataset}/preprocessed/split_train_{client_id}.csv'

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

# desired_columns = [
#         "original_shape2D_Elongation",
#         "original_shape2D_MajorAxisLength",
#         "original_shape2D_MaximumDiameter",
#         "original_shape2D_Sphericity",
#         "original_firstorder_Mean",
#         "original_firstorder_Maximum"
# ]

# desired_columns = [
#         "original_shape2D_Elongation",
#         "original_shape2D_MinorAxisLength",
#         "original_shape2D_Perimeter",
#         "original_firstorder_Median",
#         "original_firstorder_Minimum",
#         "original_firstorder_Range",
#         "original_firstorder_Mean",
#         "original_firstorder_Uniformity"
# ]

num_clients = 3
num_features = 14
num_fuzzy_sets = 5

# ## CENTRALIZED
X_train_path = "../datasets/RMI_demo/preprocessed/train.csv"
X_test_path = "../datasets/RMI_demo/preprocessed/test.csv"

X_test = pd.read_csv(X_test_path)
X_train = pd.read_csv(X_train_path)

y_train = X_train["Classe"]
X_train = X_train[desired_columns]

y_test = X_test["Classe"]
X_test = X_test[desired_columns]

# frbc_centralized = FRBC_no_opt({
#     'num_fs': num_fuzzy_sets,
#     'num_features': num_features,
#     'unique_labels': [1,2,3]
# })


# frbc_centralized.fit(X_train.values, y_train.values)
# # predictions_with_rules = frbc_centralized.predict_and_get_rule_gpu(X_test.values, "CF")
# # y_pred = [pred_class for pred_class, _ in predictions_with_rules]
# # y_pred = np.array(y_pred)
# # y_pred = y_pred + 1
# # rules = [rule_idx for _, rule_idx in predictions_with_rules]
# # print(y_pred)
# y_pred_train = frbc_centralized.predict_and_get_rule(X_test.values, "CF")
# y_pred = np.array(y_pred_train, dtype=object)[:,0]
# y_pred_clean = pd.Series(y_pred).astype(int).to_numpy()
# centr_report = classification_report(y_true = y_test, y_pred = y_pred_clean, output_dict=True)
# print("centralized")
# print(centr_report)
# quit()
plan_path = './plans/plan_federated_frbc_RMI_demo.json'

fl_plan = load_plan(plan_path, server_class=FederatedFRBCServer, client_class=FederatedFRBCClient)

clients = []
for client in range(num_clients):
    X_train, y_train = read_client_RMI_training_data(client, dataset_name, desired_columns)
    # df_X_train = pd.DataFrame(data=X_train)
    # df_y_train = pd.DataFrame(data=y_train, columns=['class'])

    clients.append(FederatedFRBCClient(type='client', id=client,
                                        X_train=X_train, y_train=y_train))


server = FederatedFRBCServer(type='server')

run_experiment(fl_plan, server, clients)


print("Experiment completed successfully.")

frbc_no_opt = FRBC_no_opt({
    'num_fs': num_fuzzy_sets,
    'num_features': num_features,
    'unique_labels': [1,2,3]
})
frbc_no_opt.load_model("./frbc_results")

# path_test_set = f'../datasets/{dataset_name}/centralized_test/centralized_test.csv'
# df_test = pd.read_csv(path_test_set)
# df_test['class'] = df_test['class'].apply(lambda x: 0 if x == 'g' else 1)
# X_test = df_test.drop(columns=['class']).to_numpy()
# y_test = df_test['class'].to_numpy()    

# y_pred = frbc_no_opt.predict_gpu(X_test.values)
y_pred_train = frbc_no_opt.predict_and_get_rule_gpu(X_test.values, "CF")

y_pred = np.array(y_pred_train, dtype=object)[:,0]
y_pred_clean = pd.Series(y_pred).astype(int).to_numpy()
print("y_pred")
print(type(y_pred_clean))


print("---------------------------")
print("test")
print(type(y_test))


print(classification_report(y_test, y_pred, output_dict=True))
