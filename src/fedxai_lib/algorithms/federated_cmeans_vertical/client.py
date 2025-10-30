# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

from typing import Final, List
from numba import jit
import numpy as np
import pandas as pd
from fedlangpy.core.entities import FedlangEntity, pickle_io

FROBENIUS_NORM: Final[str] = 'fro'


@jit(nopython=True)
def numba_norm(u: np.ndarray, v: np.ndarray):
    return np.linalg.norm(u - v)


def norm_fro(u: np.ndarray):
    return np.linalg.norm(u, ord=FROBENIUS_NORM)


class FederatedVerticalCMClient(FedlangEntity):

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

    def _compute_distances(self) -> List:
        centers = self.centers
        dist = lambda x, c: sum([(x_i - c[i])**2 for i, x_i in enumerate(x)])
        distance_2_centers = lambda x: [dist(x, center) for center in centers]
        return [distance_2_centers(instance) for instance in self.dataset]

    @pickle_io
    def update_local_centers(self, server_data) -> List:
        q_pts, pts_by_cluster = server_data
        if FederatedVerticalCMClient.get_current_round() > 1:
            cluster_ids = set(q_pts)
            centers = self.centers
            dataset = self.dataset
            for cluster_id in cluster_ids:
                mask = list(map(lambda y: y == cluster_id, q_pts))
                centers[cluster_id] = sum(dataset[mask, :]) / pts_by_cluster[cluster_id] * 1.0
        return self._compute_distances()

FederatedVerticalCMClient(type='client')
