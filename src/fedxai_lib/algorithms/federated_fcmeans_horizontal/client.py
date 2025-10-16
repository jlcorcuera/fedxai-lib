#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
  @Filename:    federated_cmeans_horizontal
  @Author:      José Luis Corcuera Bárcena
  @Time:        7/4/25 1:46 PM
"""
from typing import Final, Tuple
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


class FederatedHorizontalFCMClient(FedlangEntity):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = kwargs.get('dataset')
        if self.dataset is not None:
            self.classes = [-1] * len(self.dataset)
            self.num_features = self.dataset.shape[1]

    def init(self, input_params=None):
        self.centroid_seed = self.parameters.get('centroid_seed')
        self.num_clusters = self.parameters.get('num_clusters')
        self.lambda_factor = self.parameters.get('lambda_factor')
        if self.dataset is None:
            df = pd.read_csv(self.parameters.get('dataset'))
            X_original = df.iloc[:, :-1].values
            Y = df.iloc[:, -1].values
            X = X_original
            self.dataset = X
            self.classes = [-1] * len(self.dataset)
            self.num_features = X.shape[1]

    @pickle_io
    def init_centers(self, input_params=None):
        seed = self.centroid_seed
        c = self.num_clusters
        np.random.seed(seed)
        centers = np.random.rand(c, self.num_features)
        return centers

    @pickle_io
    def fetch_stats(self, input_params=None):
        return {
            "id": self.id,
            "num_pts": len(self.dataset)
        }

    @pickle_io
    def evaluate_cluster_assignment(self, centers) -> Tuple:
        # (1) some initialization
        num_clusters = len(centers)
        dataset = self.dataset
        num_objects = len(dataset)
        lambda_factor = self.lambda_factor
        distance_fn = numba_norm
        num_features = self.num_features

        ws = [[0] * num_features for i in range(num_clusters)]
        u = [0] * num_clusters
        u_x_c = list()
        d_x_c = list()

        for i in range(num_objects):

            denom = 0
            numer = [0] * num_clusters
            x = dataset[i]

            membership_c = list()

            for c in range(num_clusters):
                vc = centers[c]
                numer[c] = (distance_fn(x, vc)) ** ((2) / (lambda_factor - 1))
                if numer[c] == 0:
                    numer[c] = np.finfo(np.float64).eps
                denom = denom + (1 / numer[c])

            d_x_c.append(numer)

            for c in range(num_clusters):
                u_c_i = (numer[c] * denom) ** -1
                ws[c] = ws[c] + (u_c_i ** lambda_factor) * x
                u[c] = u[c] + (u_c_i ** lambda_factor)
                membership_c.append(u_c_i)

            u_x_c.append(membership_c)

        return u, ws

    @pickle_io
    def save_model(self, cluster_centers):
        self.cluster_centers = cluster_centers

FederatedHorizontalFCMClient(type='client')
