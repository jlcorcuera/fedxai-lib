# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import Final, List
from functools import reduce
from fedlangpy.core.entities import FedlangEntity, pickle_io
from numba import jit, njit


FROBENIUS_NORM: Final[str] = 'fro'


def norm_fro(u: np.ndarray):
    return np.linalg.norm(u, ord=FROBENIUS_NORM)

@njit
def compute_u_c_t_matrix_numba(distance_matrix, lambda_factor):
    mem_obj_cluster = lambda idx, C: reduce(lambda a, b: a + b,
                                            [((C[idx] / C[i]) if idx != i else 1.0) ** (2.0 / (lambda_factor - 1)) for i
                                             in range(len(C))]) ** -1
    mem_obj_clusters = lambda C_param: [mem_obj_cluster(idx, C_param) for idx in range(len(C_param))]
    return [mem_obj_clusters(obj) for obj in distance_matrix]

@njit
def compute_distances_numba(n_features: int, centers: np.array, dataset: np.array):
    n_centers = len(centers)
    n_obj = len(dataset)
    all_distances = [[0.0] * n_centers for i in range(n_obj)]
    for idx in range(n_obj):
        instance = dataset[idx]
        for idx_center in range(n_centers):
            center = centers[idx_center]
            distance = 0
            for i in range(n_features):
                distance = distance + (instance[i] - center[i]) ** 2
            all_distances[idx][idx_center] = distance
    return all_distances


@njit
def compute_distance_matrix_numba(n_centers: int, n_objects: int, client_responses):
    object_cluster_distance_mtrx = [[0.0] * n_centers for i in range(n_objects)]
    n_clients = len(client_responses)
    for i in range(n_objects):
        for j in range(n_centers):
            distance = 0
            for k in range(n_clients):
                distance = distance + client_responses[k][i][j]
            object_cluster_distance_mtrx[i][j] = distance ** 0.5
    return object_cluster_distance_mtrx


class FederatedVerticalFCMServer(FedlangEntity):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self, input_params=None):
        self.epsilon = self.parameters.get("epsilon")
        self.norm_fn = norm_fro
        self.max_number_rounds = self.parameters.get("max_number_rounds")
        self.fnorms = []
        self.num_clusters = self.parameters.get("num_clusters")
        self.lambda_factor = self.parameters.get("lambda_factor")
        self.u_c_t = None
        self.last_D_matrix = None

    def stop_federation(self):
        current_round = FederatedVerticalFCMServer.get_current_round()
        if current_round > 1:
            if self.fnorms[-1] < self.epsilon:
                return True

        return current_round >= self.max_number_rounds

    @pickle_io
    def get_cluster_stats(self, input_params=None):
        return self.u_c_t

    @pickle_io
    def process_round(self, client_responses: List):
        lambda_factor = self.lambda_factor

        n_objects = len(client_responses[0])
        client_responses = np.array(client_responses)
        distance_matrix = np.array(compute_distance_matrix_numba(self.num_clusters, n_objects, client_responses))

        if self.last_D_matrix is not None:
            d_t_1 = self.last_D_matrix
            fnorm_value = self.norm_fn(distance_matrix - d_t_1)
            self.fnorms.append(fnorm_value)

        self.u_c_t = compute_u_c_t_matrix_numba(distance_matrix, lambda_factor)
        self.last_D_matrix = distance_matrix

FederatedVerticalFCMServer(type='server')
