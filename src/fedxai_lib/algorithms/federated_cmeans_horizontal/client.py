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


class FederatedHorizontalCMClient(FedlangEntity):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = kwargs.get('dataset')
        if self.dataset is not None:
            self.classes = [-1] * len(self.dataset)
            self.num_features = self.dataset.shape[1]

    def init(self, input_params=None):
        self.centroid_seed = self.parameters.get('centroid_seed')
        self.num_clusters = self.parameters.get('num_clusters')
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
    def evaluate_cluster_assignment(self, centers) -> List:
        # (1). some initialization
        num_clusters = len(centers)
        dataset = self.dataset
        num_features = self.num_features
        classes = self.classes
        num_objects = len(dataset)
        get_label = self._get_label

        nc_list = [0] * num_clusters
        lsc_list = [np.array([0] * num_features) for i in range(num_clusters)]

        # (2) Updating the class value for each object in the dataset
        for i in range(num_objects):
            obj = dataset[i]
            label = get_label(obj, centers)
            classes[i] = label
            # updating stats for each cluster
            nc_list[label] = nc_list[label] + 1
            lsc_list[label] = lsc_list[label] + obj

        for i in range(num_clusters):
            if nc_list[i] == 0:
                nc_list[i] = 0
                lsc_list[i] = 0

        # (3) Preparing data to return
        to_return = [(lsc_list[i], nc_list[i]) for i in range(num_clusters)]
        return to_return

    def _get_label(self, obj_data: np.array, centers: List[np.array]):

        max_value = 2 ** 64
        num_clusters = len(centers)
        distance_fn = numba_norm
        label_idx = -1

        for i in range(num_clusters):
            center = centers[i]
            distance = distance_fn(obj_data, center)

            if distance < max_value:
                label_idx = i
                max_value = distance

        return label_idx

    @pickle_io
    def save_model(self, cluster_centers):
        self.cluster_centers = cluster_centers

FederatedHorizontalCMClient(type='client')
