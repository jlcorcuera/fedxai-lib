# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

"""
Created on Aug 03 12:25 p.m. 2023

@author: jose 
"""
import os
import numpy as np
from typing import Dict, List, Tuple
import pickle
from fedxai_lib.algorithms.federated_frt.model import FedFRTModel
from fedxai_lib.algorithms.federated_frt.node import FedFRTNode
from fedxai_lib.algorithms.federated_frt.utils.custom_logger import logger_info
from fedxai_lib.algorithms.federated_frt.utils.fuzzy_sets_utils import create_fuzzy_sets_from_strong_partition
from fedxai_lib.algorithms.federated_frt.utils.stats_utils import fuzzy_gain, fuzzy_variance
from fedlangpy.core.entities import FedlangEntity, pickle_io


class FedFRTServer(FedlangEntity):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        FedFRTNode.reset_node_id()
        self.v_dict: Dict = None
        self.root_node: FedFRTNode = None
        self.model = None

    def init(self, input_params=None):
        self.features_names = self.parameters.get("features_names")
        self.target_feature = self.parameters.get("target")
        self.num_features = len(self.features_names)
        self.features = [i for i in range(self.num_features)]
        self.num_fuzzy_sets = self.parameters.get("num_fuzzy_sets")
        self.gain_threshold = self.parameters.get("gain_threshold")
        self.max_depth = self.parameters.get("max_depth")
        self.max_number_rounds = self.parameters.get("max_number_rounds")
        self.min_samples_split_ratio = self.parameters.get("min_samples_split_ratio")
        self.model_output_file = self.parameters.get("model_output_file")

        splits = np.linspace(0, 1, self.num_fuzzy_sets, dtype=float)
        fuzzy_sets = []
        for idx_feature in range(self.num_features):
            fuzzy_sets.append(create_fuzzy_sets_from_strong_partition(splits))

        self.fuzzy_sets = fuzzy_sets

    @pickle_io
    def init_frt(self, root_node_clients_stats):
        root_node_num_samples = sum([client_stats.WS for client_stats in root_node_clients_stats])
        root_node_variance = fuzzy_variance(root_node_clients_stats)
        self.min_samples_split: int = int(self.min_samples_split_ratio * root_node_num_samples)
        self.root_node: FedFRTNode = FedFRTNode(variance=root_node_variance,
                                                num_samples=root_node_num_samples,
                                                num_samples_above_thr=root_node_num_samples,
                                                mean_activation_force=1.0)
        self.v_dict: Dict = dict()
        self.v_dict[1] = [(self.root_node, list(self.features))]

    def build_model(self, input_params=None):
        self.model: FedFRTModel = FedFRTModel(self.features_names, self.root_node)
        return self.model

    def stop_tree_growing(self) -> bool:
        current_round = FedFRTServer.get_current_round()
        next_round = current_round + 1
        v_t_1 = self.get_v(next_round)
        return current_round >= self.max_number_rounds or not v_t_1

    @pickle_io
    def get_current_level_stats(self, input_params=None) -> Tuple:
        round_t = FedFRTServer.get_current_round()
        depth = round_t - 1
        mem_threshold = 0.5 ** (depth + 1)
        v_t = self.get_v(round_t)
        return v_t, mem_threshold

    def get_model(self) -> FedFRTModel:
        return self.model

    def get_v(self, p_round: int) -> List[Tuple]:
        return self.v_dict.get(p_round)

    def get_root_node(self) -> FedFRTNode:
        return self.root_node

    @pickle_io
    def grow_tree(self, clients_stats: List) -> None:
        current_round = FedFRTServer.get_current_round()
        next_round = current_round + 1
        depth = current_round - 1
        v_t_1 = list()
        for node_q, h_q in self.v_dict.get(current_round):
            logger_info(f'depth = {depth}, node_id = {node_q.get_comp_id()} variance = {node_q.variance}, gain = {node_q.gain}')
            max_depth_validation = self.max_depth is not None and self.max_depth == depth
            min_num_split_validation = node_q.num_samples_above_thr < self.min_samples_split
            logger_info(f'max_depth_validation = {max_depth_validation}, min_num_split_validation = {min_num_split_validation}')
            if max_depth_validation or min_num_split_validation:
                node_q.mark_as_leaf()
                continue

            node_variance = node_q.variance

            best_gain = None
            best_feature = None

            for feature in h_q:
                if len(self.fuzzy_sets[feature]):
                    current_gain = fuzzy_gain(self.num_fuzzy_sets, node_variance, node_q.id, feature, clients_stats)
                    logger_info(f'feature = {feature}, current_gain = {current_gain}')
                    if (best_gain is None or current_gain > best_gain) and current_gain != node_variance:
                        best_gain = current_gain
                        best_feature = feature
                else:
                    logger_info(f"feature = {feature} don't have fuzzy-sets.")

            print(f'depth = {depth}, best_feature = {best_feature}, best_gain = {best_gain}, node_variance = {node_variance}')

            if best_gain and best_gain >= self.gain_threshold:
                new_h_q = list(h_q)
                new_h_q.pop(new_h_q.index(best_feature))
                was_added_nodes = False
                for fs_idx, fs in enumerate(self.fuzzy_sets[best_feature]):
                    data_owner_sufficient_points = sum([stat[node_q.id][best_feature][fs_idx].S for stat in clients_stats]) >= self.num_features + 1
                    if data_owner_sufficient_points:
                        fs_node_variance = fuzzy_variance([stat[node_q.id][best_feature][fs_idx] for stat in clients_stats])
                        fs_activation_forces = sum([stat[node_q.id][best_feature][fs_idx].WS for stat in clients_stats])
                        fs_node_samples = sum([stat[node_q.id][best_feature][fs_idx].S for stat in clients_stats])
                        fs_node_samples_above_thr = sum([stat[node_q.id][best_feature][fs_idx].NTH for stat in clients_stats])
                        logger_info(f'current_round = {current_round}, fs_node_samples_above_thr = {fs_node_samples_above_thr}')
                        mean_activation_force = fs_activation_forces / fs_node_samples
                        node_q_b_f_t = FedFRTNode(feature=best_feature,
                                                  feature_name=self.features_names[best_feature],
                                                  f_set=fs,
                                                  gain=best_gain,
                                                  depth=current_round,
                                                  parent=node_q,
                                                  variance=fs_node_variance,
                                                  num_samples=fs_node_samples,
                                                  num_samples_above_thr=fs_node_samples_above_thr,
                                                  mean_activation_force=mean_activation_force)
                        node_q.add_children(node_q_b_f_t)
                        was_added_nodes = True
                        if new_h_q:
                            v_t_1.append((node_q_b_f_t, new_h_q))
                        else:
                            node_q_b_f_t.mark_as_leaf()
                    else:
                        logger_info(f'Feature = {best_feature}, fs_idx = {fs_idx}, no enough points.')
                if not was_added_nodes:
                    node_q.mark_as_leaf()
            else:
                node_q.mark_as_leaf()
        self.v_dict[next_round] = v_t_1

    @pickle_io
    def get_antecedents_nodes_form(self, input_params=None):
        return self.model.get_rules()

    @pickle_io
    def compute_consequents(self, client_responses):
        antecedents = self.model.get_rules()
        consequents = []
        for i in range(len(antecedents)):
            sum_dot_x_t_x_list = np.add.reduce([dot_x_t_x_list[i][0] for dot_x_t_x_list in client_responses])
            sum_dot_x_t_y_list = np.add.reduce([dot_x_t_y_list[i][1] for dot_x_t_y_list in client_responses])

            new_consequents, residuals, rank, s = np.linalg.lstsq(sum_dot_x_t_x_list, sum_dot_x_t_y_list)
            #logger_info(f'RULE {i}, rank = {rank}, residuals = {residuals}, s = {s}')
            w_0 = new_consequents[-1]
            new_consequents = np.insert(new_consequents, 0, w_0, axis=0)
            new_consequents = np.delete(new_consequents, -1, axis=0)
            consequents.append(new_consequents)
        consequents = np.array(consequents)
        self.model.set_consequents(consequents)
        self.model.set_num_samples_by_rule([sum([dot_x_t_y_list[i][2] for dot_x_t_y_list in client_responses]) for i in range(len(antecedents))])

    @pickle_io
    def get_consequents(self, input_params=None):
        return self.model.get_consequents()

    @pickle_io
    def compute_rule_error_stats(self, client_responses):
        antecedents = self.model.get_rules()
        max_ae = None
        min_ae = None
        for client_response in client_responses:
            max_ae_client, min_ae_client = client_response
            if max_ae is None or max_ae < max_ae_client:
                max_ae = max_ae_client
            if min_ae is None or min_ae > min_ae_client:
                min_ae = min_ae_client

        delta_max_min_ae = max_ae - min_ae
        num_of_rules = len(antecedents)
        return num_of_rules, min_ae, delta_max_min_ae

    @pickle_io
    def compute_rule_weights(self, conf_supp_partial_data_list):
        antecedents = self.model.get_rules()
        num_of_rules = len(antecedents)
        rule_weight = np.zeros(num_of_rules)
        for rule_idx in range(num_of_rules):
            sum_weighted_membership_value = sum([client_data[rule_idx][0] for client_data in conf_supp_partial_data_list])
            sum_firing_strength = sum([client_data[rule_idx][1] for client_data in conf_supp_partial_data_list])
            num_of_training_samples = sum([client_data[rule_idx][2] for client_data in conf_supp_partial_data_list])

            fuzzy_confidence = (sum_weighted_membership_value / sum_firing_strength) if sum_firing_strength != 0 else 0
            fuzzy_support = sum_weighted_membership_value / num_of_training_samples
            if fuzzy_support + fuzzy_confidence != 0:
                rule_weight[rule_idx] = (2 * fuzzy_support * fuzzy_confidence) / (fuzzy_support + fuzzy_confidence)
            else:
                rule_weight[rule_idx] = 0
        self.model.set_rule_weights(rule_weight)

    def save_model(self, input_params=None):
        output_file = self.model_output_file
        logger_info(f'model_output_file = {output_file}')
        with open(output_file, 'wb') as model_file:
            pickle.dump(self.model, model_file, pickle.HIGHEST_PROTOCOL)
        return self.model

FedFRTServer(type="server")
