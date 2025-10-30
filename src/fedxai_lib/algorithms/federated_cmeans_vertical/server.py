# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import Final, List
from functools import reduce
from fedlangpy.core.entities import FedlangEntity, pickle_io
from collections import Counter


FROBENIUS_NORM: Final[str] = 'fro'


def norm_fro(u: np.ndarray):
    return np.linalg.norm(u, ord=FROBENIUS_NORM)


class FederatedVerticalCMServer(FedlangEntity):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self, input_params=None):
        self.epsilon = self.parameters.get("epsilon")
        self.norm_fn = norm_fro
        self.max_number_rounds = self.parameters.get("max_number_rounds")
        self.D_matrix = []
        self.fnorms = []
        self.num_clusters = self.parameters.get("num_clusters")
        self.q_pts = None
        self.pts_by_cluster = None

    def stop_federation(self):
        current_round = FederatedVerticalCMServer.get_current_round()
        if current_round > 1:
            D_matrix = self.D_matrix
            d_t = np.array(D_matrix[-1])
            d_t_1 = np.array(D_matrix[-2])
            fnorm_value = self.norm_fn(d_t - d_t_1)
            self.fnorms.append(fnorm_value)
            if fnorm_value > self.epsilon:
                return True

        return current_round >= self.max_number_rounds

    @pickle_io
    def get_cluster_stats(self, input_params=None):
        return self.q_pts, self.pts_by_cluster

    @pickle_io
    def process_round(self, client_responses: List) -> (bool, List, List):
        D_matrix = self.D_matrix
        distance_matrices = list(map(lambda ar: np.matrix(ar), client_responses))
        object_cluster_distance = reduce(lambda a, b: a + b, distance_matrices)
        distance_matrix = list(map(lambda d: [i**0.5 for i in d], object_cluster_distance.tolist()))
        D_matrix.append(distance_matrix)

        q_pts = [np.argmin(distances) for distances in distance_matrix]
        counter = Counter(q_pts)
        pts_by_cluster = []
        for cluster in range(self.num_clusters):
            pts_by_cluster.append(counter.get(cluster, 0))

        self.q_pts = q_pts
        self.pts_by_cluster = pts_by_cluster

FederatedVerticalCMServer(type='server')