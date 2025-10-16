#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 03 12:25 p.m. 2023

@author: jose 
"""
from typing import Final
import pandas as pd
import numpy as np
from typing import Callable, List, Tuple
import pickle
from fedxai_lib.algorithms.federated_frt.node import NODE_ROOT_ID, FedFRTNode, FedFRTFeatureFSStats
from fedxai_lib.algorithms.federated_frt.utils.custom_logger import logger_info
from fedxai_lib.algorithms.federated_frt.utils.fuzzy_sets_utils import t_norm, create_fuzzy_sets_from_strong_partition
from fedxai_lib.algorithms.federated_frt.utils.stats_utils import compute_weight_rules_step1, compute_weight_rules_step2, \
    compute_firing_strengths
from fedlangpy.core.entities import FedlangEntity, pickle_io

np_square: Callable = np.square

TARGET: Final[str] = 'framesDisplayed_H'
RUN_COLUMN: Final[str] = 'run'
UE_COLUMN: Final[str] = 'ue'


class FedFRTClient(FedlangEntity):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tmp_membership_degree = dict()
        self.tmp_row_vector = dict()

        self.scaler_X = kwargs.get("scaler_X")
        self.scaler_y = kwargs.get("scaler_y")
        self.X_train = kwargs.get("X_train")
        self.y_train = kwargs.get("y_train")
        self.X_test = kwargs.get("X_test")
        self.y_test = kwargs.get("y_test")

        self.firing_strengths = None
        self.total_number_obfuscation_check = 0
        self.total_number_obfuscation_applied_case1 = 0
        self.total_number_obfuscation_applied_case2 = 0
        self.total_number_obfuscation_applied_case3 = 0

    def init(self, input_params=None):
        self.num_fuzzy_sets = self.parameters.get("num_fuzzy_sets")
        self.features_names = self.parameters.get("features_names")
        self.model_output_file = self.parameters.get("model_output_file")
        self.num_features = len(self.features_names)
        splits = np.linspace(0, 1, self.num_fuzzy_sets, dtype=float)
        fuzzy_sets = []
        for idx_feature in range(self.num_features):
            fuzzy_sets.append(create_fuzzy_sets_from_strong_partition(splits))
        self.fuzzy_sets = fuzzy_sets
        self.obfuscate = self.parameters.get("obfuscate", True)

        if self.X_train is None:
            self.X_train = pd.read_csv(self.parameters.get('dataset_X_train'))
            self.y_train = pd.read_csv(self.parameters.get('dataset_y_train'))
            self.X_test = pd.read_csv(self.parameters.get('dataset_X_test'))
            self.y_test = pd.read_csv(self.parameters.get('dataset_X_test'))


        self.X_train = self.X_train.to_numpy()
        self.y_train = self.y_train.values.reshape(-1, 1).ravel()
        self.X_test = self.X_test.to_numpy()
        self.y_test = self.y_test.values.reshape(-1, 1).ravel()

        self.training_set = np.concatenate((self.X_train, self.y_train.reshape(self.y_train.shape[0], 1)), axis=1)
        self.tmp_row_vector[NODE_ROOT_ID] = np.array([idx for idx in range(self.training_set.shape[0])])
        self.tmp_membership_degree[NODE_ROOT_ID] = np.ones(self.training_set.shape[0])


    @pickle_io
    def get_stats_for_root_node(self, input_params=None) -> FedFRTFeatureFSStats:
        membership_degree = self.tmp_membership_degree.get(NODE_ROOT_ID)
        rows = self.training_set
        WSS = (np_square(rows[:, -1]) * membership_degree).sum()
        WLS = (rows[:, -1] * membership_degree).sum()
        WS = membership_degree.sum()
        return FedFRTFeatureFSStats(WSS=WSS, WLS=WLS, WS=WS)

    @pickle_io
    def compute_stats(self, stats_from_server):
        round_t = FedFRTClient.get_current_round()
        v_t, mem_threshold = stats_from_server
        tmp_membership_degree = dict()
        tmp_row_vector = dict()
        logger_info(f"round_t = {round_t}, client_id = {self.id} " + "=" * 10)
        results = dict()
        for node_q, h_q in v_t:
            stats = dict()
            node_id = node_q.id
            node_comp_id = node_q.get_comp_id()
            membership_degree = self.tmp_membership_degree.get(node_comp_id)
            mask_rows = self.tmp_row_vector.get(node_comp_id)
            rows = self.training_set
            logger_info(f'node_q = {node_q.get_comp_id()}, h_q = {h_q}')
            for feature in h_q:
                stats_feature = list()
                fuzzy_sets = self.fuzzy_sets[feature]
                row_vector, membership_vector, mask_idx = self._multidivide(rows, membership_degree, mask_rows, feature)
                for i_fs in range(len(fuzzy_sets)):
                    current_fuzzy_set = fuzzy_sets[i_fs]
                    fs_row_vector = row_vector[i_fs]
                    fs_membership_vector = membership_vector[i_fs]
                    fs_mask_idx = mask_idx[i_fs]
                    if fs_row_vector.shape[0] > 0:
                        WSS = (np_square(fs_row_vector[:, -1]) * fs_membership_vector).sum()
                        WLS = (fs_row_vector[:, -1] * fs_membership_vector).sum()
                        WS = fs_membership_vector.sum()
                        S = len(fs_membership_vector)
                        NTH = sum([1 for mem in fs_membership_vector if mem >= mem_threshold])
                        stats_feature.append(FedFRTFeatureFSStats(WSS=WSS, WLS=WLS, WS=WS, S=S, NTH=NTH))
                    else:
                        stats_feature.append(FedFRTFeatureFSStats(all_zeros=True))
                    tmp_key = f'{node_id}_{feature}_{current_fuzzy_set.get_term()}'
                    tmp_row_vector[tmp_key] = fs_mask_idx
                    tmp_membership_degree[tmp_key] = fs_membership_vector
                stats[feature] = stats_feature
            results[node_id] = stats

        self.tmp_membership_degree = {**self.tmp_membership_degree, **tmp_membership_degree}
        t1 = len(self.tmp_row_vector)
        self.tmp_row_vector = {**self.tmp_row_vector, **tmp_row_vector}
        assert t1 + len(tmp_row_vector) == len(self.tmp_row_vector)
        return self.__check_and_nullify(v_t, results)

    def __check_and_nullify(self, v_t, results):
        if self.obfuscate:
            for node_q, h_q in v_t:
                node_id = node_q.id
                feature_stats_list = results[node_id]
                for feature in h_q:
                    feature_stats = feature_stats_list.get(feature)
                    for i_fs in range(len(feature_stats)):
                        self.total_number_obfuscation_check = self.total_number_obfuscation_check + 1
                        feature_fs_stats:FedFRTFeatureFSStats = feature_stats[i_fs]
                        if 1 <= feature_fs_stats.S <= 2:
                            self.total_number_obfuscation_applied_case1 = self.total_number_obfuscation_applied_case1 + 1
                            feature_stats[i_fs] = FedFRTFeatureFSStats(all_zeros=True)
                            logger_info(f'STATS NODE, IN OBFUSCATE CASE 1 client_id = {self.id}, feature = {feature}, i_fs = {i_fs}, WS = {feature_fs_stats.WS}, S = {feature_fs_stats.S}')
                        elif node_q.get_comp_id() == NODE_ROOT_ID and feature_fs_stats.WS == feature_fs_stats.S:
                            self.total_number_obfuscation_applied_case3 = self.total_number_obfuscation_applied_case3 + 1
                            feature_stats[i_fs] = FedFRTFeatureFSStats(all_zeros=True)
                            logger_info(f'STATS ROOT NODE, IN OBFUSCATE client_id = {self.id}, feature = {feature}, i_fs = {i_fs}, WS = {feature_fs_stats.WS}, S = {feature_fs_stats.S}')
                        else:
                            WS_j_prev = None if i_fs == 0 else feature_stats[i_fs - 1].WS
                            WS_j_next = feature_stats[i_fs + 1].WS if i_fs < len(feature_stats) - 1 else None
                            if feature_fs_stats.WS > 0 and (WS_j_prev is None or WS_j_prev == 0) and (WS_j_next is None or WS_j_next == 0):
                                self.total_number_obfuscation_applied_case2 = self.total_number_obfuscation_applied_case2 + 1
                                feature_stats[i_fs] = FedFRTFeatureFSStats(all_zeros=True)
                                logger_info(f'STATS NODE, IN OBFUSCATE client_id = {self.id}, feature = {feature}, i_fs = {i_fs}, WS = {feature_fs_stats.WS}, S = {feature_fs_stats.S}')
        return results

    def get_privacy_stats(self):
        data = {
            'client_id': [self.id],
            'total_number_obfuscation_check': [self.total_number_obfuscation_check],
            'total_number_obfuscation_applied_case1': [self.total_number_obfuscation_applied_case1],
            'total_number_obfuscation_applied_case2': [self.total_number_obfuscation_applied_case2],
            'total_number_obfuscation_applied_case3': [self.total_number_obfuscation_applied_case3]
        }
        return pd.DataFrame(data)

    def _multidivide(self, rows, membership, mask_rows, feature) -> Tuple:
        assert len(mask_rows) == len(membership)
        mem_vect = list()
        row_vect = list()
        mask_vect = list()
        if rows.shape[0] == 0:
            for fuzzy_set in self.fuzzy_sets[feature]:
                row_vect.append(np.array([]))
                mem_vect.append(np.array([]))
                mask_vect.append(np.array([]))
        else:
            for fuzzy_set in self.fuzzy_sets[feature]:
                mask_idx_rows = np.array([idx for idx in mask_rows if fuzzy_set.get_value(rows[idx][feature]) != 0])
                if mask_idx_rows.shape[0] > 0:
                    filtered_rows = rows[mask_idx_rows, :]
                    mask_idx_membership = np.array([i for i in range(len(mask_rows)) if fuzzy_set.get_value(rows[mask_rows[i]][feature]) != 0])
                    activation_force = membership[mask_idx_membership]
                    row_vect.append(filtered_rows)
                    mem_vect.append(t_norm(np.array(list(map(lambda x: fuzzy_set.get_value(x), filtered_rows[:, feature]))), activation_force))
                    mask_vect.append(mask_idx_rows)
                else:
                    row_vect.append(np.array([]))
                    mem_vect.append(np.array([]))
                    mask_vect.append(np.array([]))
        return row_vect, mem_vect, mask_vect

    @pickle_io
    def compute_matrices_for_consequents(self, rules: List[List[FedFRTNode]]):
        self.antecedents = rules
        X_train = self.X_train
        y_train = self.y_train
        leave_nodes = [rule[-1] for rule in rules]
        cache_activation_force = {node.get_comp_id(): node.mean_activation_force for node in leave_nodes}
        cfs: Callable = lambda x: compute_firing_strengths(x, rules, cache_activation_force)
        firing_strengths_list = list(map(cfs, self.X_train))
        firing_strengths = np.array(firing_strengths_list)

        X_train = X_train.copy()
        y_train = y_train.copy()

        f = firing_strengths.copy()
        mf, nf = f.shape

        u = np.unique(X_train[:, -1])
        if u.shape[0] != 1 or u[0] != 1:
            X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))

        results = []

        for i in range(0, nf):
            n = leave_nodes[i]
            # Select firing strength of the selected rule
            _w = f[:, i]

            mask_rows = np.array(list(map(lambda _w_x: _w_x > 0, _w)))

            w = _w[mask_rows]
            _x = X_train[mask_rows, :]
            _y = y_train[mask_rows]

            # Weight input with firing strength
            xw = _x * np.sqrt(w[:, np.newaxis])

            # Weight output with firing strength
            yw = _y * np.sqrt(w)

            dot_x_t_x = np.dot(xw.T, xw)
            dot_x_t_y = np.dot(xw.T, yw)

            results.append((dot_x_t_x, dot_x_t_y, len(self.tmp_row_vector.get(n.get_comp_id(), []))))
        return results

    @pickle_io
    def compute_rule_weights_step1(self, consequents):
        antecedents = self.antecedents
        X_train = self.X_train
        y_train = self.y_train
        firing_strengths, absolute_error_matrix, max_ae, min_ae = compute_weight_rules_step1(antecedents,
                                                                                             consequents,
                                                                                             X_train,
                                                                                             y_train)
        self.firing_strengths = firing_strengths
        self.absolute_error_matrix = absolute_error_matrix
        return max_ae, min_ae

    @pickle_io
    def compute_weight_rules_step2(self, rules_error_stats) -> List:
        num_of_rules, min_ae, delta_max_min_ae = rules_error_stats
        return compute_weight_rules_step2(self.firing_strengths,
                                          len(self.X_train),
                                          num_of_rules,
                                          self.absolute_error_matrix,
                                          min_ae,
                                          delta_max_min_ae)

    def evaluate_model(self, iteration, for_qoe: bool = False):
        X_train = None
        y_train = None
        X_test = None
        y_test = None
        if not for_qoe:
            X_train = self.X_train
            y_train = self.y_train
            X_test = self.X_test
            y_test = self.y_test
        else:
            df_train_original = self.df_train_original
            df_test_ue_original = self.df_test_ue_original
            if df_train_original is not None:
                X_train = df_train_original.drop(columns=[TARGET]).to_numpy()
                y_train = df_train_original[TARGET].ravel()
            if df_test_ue_original is not None:
                X_test = df_test_ue_original.drop(columns=[TARGET]).to_numpy()
                y_test = df_test_ue_original[TARGET].ravel()

        model = self._model
        client_id = self.client_id

        data = []
        data_append = data.append

        if X_train is not None:
            for input_vector_t, y_true in zip(X_train, y_train):
                input_vector = input_vector_t[2:] if for_qoe else input_vector_t
                result, rule_id, t_num_rules = model.predict(np.array([input_vector]))[0]
                if for_qoe:
                    data_append([input_vector_t[0], input_vector_t[1], iteration, 'train', rule_id, t_num_rules, y_true, result])
                else:
                    data_append([client_id, iteration, 'train', rule_id, t_num_rules, y_true, result])

        if X_test is not None:
            for input_vector_t, y_true in zip(X_test, y_test):
                input_vector = input_vector_t[2:] if for_qoe else input_vector_t
                result, rule_id, t_num_rules = model.predict(np.array([input_vector]))[0]
                if for_qoe:
                    data_append([input_vector_t[0], input_vector_t[1], iteration, 'test', rule_id, t_num_rules, y_true, result])
                else:
                    data_append([client_id, iteration, 'test', rule_id, t_num_rules, y_true, result])

        if for_qoe:
            columns = ['client_id', 'run', 'fold_iteration', 'type', 'rule_id', 'num_rules_activated', 'true_value', 'predicted_value']
        else:
            columns = ['client_id', 'fold_iteration', 'type', 'rule_id', 'num_rules_activated', 'true_value', 'predicted_value']
        to_return_df = pd.DataFrame(data, columns=columns)
        scaler_y = self.scaler_y
        to_return_df['true_value'] = scaler_y.inverse_transform(to_return_df['true_value'].values.reshape(-1, 1))
        to_return_df['predicted_value'] = scaler_y.inverse_transform(to_return_df['predicted_value'].values.reshape(-1, 1))
        return to_return_df

    def save_model(self, input_params=None):
        final_model = input_params
        logger_info(f'Receiving final model: {type(type(final_model))}')
        output_file = self.model_output_file
        model_client_id = output_file.replace(".", f"_client_{self.id}.")
        logger_info(f'model_output_file = {model_client_id}')
        with open(model_client_id, 'wb') as model_file:
            pickle.dump(final_model, model_file, pickle.HIGHEST_PROTOCOL)


FedFRTClient(type="client")