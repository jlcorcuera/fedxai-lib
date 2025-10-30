# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

from typing import Final, List
from numba import jit, njit
import numpy as np
import pandas as pd
from fedlangpy.core.entities import FedlangEntity, pickle_io

FROBENIUS_NORM: Final[str] = 'fro'


@jit(nopython=True)
def numba_norm(u: np.ndarray, v: np.ndarray):
    return np.linalg.norm(u - v)


def norm_fro(u: np.ndarray):
    return np.linalg.norm(u, ord=FROBENIUS_NORM)


@njit
def update_local_centers_numba(centers: np.array, n_features: int, dataset: np.array, u_t: np.array,
                               lambda_factor: float):
    n_centers = len(centers)
    total_numerators = [[0.] * n_features for i in range(n_centers)]
    total_denominators = [0.] * n_centers
    for i in range(len(dataset)):
        obj = dataset[i]
        for cluster_id in range(n_centers):
            tmp_prod = [obj[j] * (u_t[i][cluster_id] ** lambda_factor) for j in range(n_features)]
            total_numerators[cluster_id] = [total_numerators[cluster_id][j] + tmp_prod[j] for j in range(n_features)]
            total_denominators[cluster_id] = total_denominators[cluster_id] + u_t[i][cluster_id] ** lambda_factor
    return [([total_numerators[i][j] / total_denominators[i] for j in range(n_features)]
             if total_denominators[i] != 0 else None)
            for i in range(n_centers)]


class FederatedVerticalFCMClient(FedlangEntity):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = kwargs.get('dataset')
        if self.dataset is not None:
            self.num_features = self.dataset.shape[1]

    def init(self, input_params=None):
        if self.dataset is None:
            df = pd.read_csv(self.parameters.get('dataset'))
            X = df.iloc[:, :-1].values
            self.dataset = X
            self.num_features = X.shape[1]
        self.num_clusters = self.parameters.get('num_clusters')
        self.centers = np.random.rand(self.num_clusters, self.num_features)
        self.lambda_factor = self.parameters.get("lambda_factor")

    def _compute_distances(self) -> List:
        centers = self.centers
        dist = lambda x, c: sum([(x_i - c[i])**2 for i, x_i in enumerate(x)])
        distance_2_centers = lambda x: [dist(x, center) for center in centers]
        return [distance_2_centers(instance) for instance in self.dataset]

    @pickle_io
    def update_local_centers(self, server_data=None) -> List:
        u_t = server_data
        if FederatedVerticalFCMClient.get_current_round() > 1:
            lambda_factor = self.lambda_factor
            centers = self.centers
            dataset = self.dataset
            tmp_centers = update_local_centers_numba(centers, self.num_features, dataset, np.array(u_t), lambda_factor)
            self.centers = [tmp_centers[i] if tmp_centers[i] is not None else centers[i] for i in range(len(centers))]
            self.centers = np.array(self.centers)
        return self._compute_distances()

FederatedVerticalFCMClient(type='client')
