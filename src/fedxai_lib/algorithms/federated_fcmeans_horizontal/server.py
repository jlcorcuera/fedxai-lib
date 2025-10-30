# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

from typing import Final, List
import numpy as np
import random
from fedlangpy.core.entities import FedlangEntity, pickle_io

FROBENIUS_NORM: Final[str] = 'fro'


def norm_fro(u: np.ndarray):
    return np.linalg.norm(u, ord=FROBENIUS_NORM)


class FederatedHorizontalFCMServer(FedlangEntity):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self, input_params):
        self.epsilon = self.parameters.get('epsilon')
        self.max_number_rounds = self.parameters.get('max_number_rounds', 10)
        self.num_clusters = self.parameters.get('num_clusters')
        self.min_num_clients = self.parameters.get('min_num_clients')
        self.norm_fm = norm_fro
        self.fnorms = []
        self.cluster_centers = []

    @pickle_io
    def register_clients(self, clients_stats):
        self.clients = clients_stats

    def stop_federation(self):
        return FederatedHorizontalFCMServer.get_current_round() >= self.max_number_rounds

    def sample_clients(self, input_params):
        num_clients = len(self.clients)
        selected_clients_idx = random.sample(range(num_clients), self.min_num_clients)
        return [self.clients[idx].get("id") for idx in selected_clients_idx]

    @pickle_io
    def get_centers(self, input_params=None) -> List:
        return self.cluster_centers[-1]

    @pickle_io
    def init_centers(self, centers: List):
        centers_one_client = centers[0]
        self.cluster_centers.append(centers_one_client)
        return centers

    def select_one_client(self, input_params=None):
        selected_clients_idx = random.choice([idx for idx in range(len(self.clients))])
        return [self.clients[selected_clients_idx].get("id")]

    @pickle_io
    def process_clustering_results(self, client_responses: List):
        num_clients = len(client_responses)
        num_clusters = self.num_clusters
        num_features = len(client_responses[0][1][0])
        u_list = [0] * num_clusters
        ws_list = [[0] * num_features for i in range(num_clusters)]

        for client_idx in range(num_clients):
            response = client_responses[client_idx]
            for i in range(num_clusters):
                client_u = response[0][i]
                client_ws = response[1][i] if response[1][i] is np.array else np.array(response[1][i])
                u_list[i] = u_list[i] + client_u
                ws_list[i] = ws_list[i] + client_ws

        new_cluster_centers = []
        prev_cluster_centers = None

        if self.cluster_centers:
            prev_cluster_centers = self.cluster_centers[-1]

        for i in range(num_clusters):
            u = u_list[i]
            ws = ws_list[i]
            if u == 0 and prev_cluster_centers:
                center = prev_cluster_centers[i]
            else:
                center = ws / u
            new_cluster_centers.append(np.array(center))

        self.cluster_centers.append(new_cluster_centers)
        return new_cluster_centers

    @pickle_io
    def save_model(self, input_params=None):
        return self.cluster_centers

FederatedHorizontalFCMServer(type='server')
