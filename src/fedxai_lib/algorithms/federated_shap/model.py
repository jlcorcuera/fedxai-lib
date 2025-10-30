# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import Union, List
import shap


class FedShapModel:

    def __init__(self, centroids: Union[List, np.ndarray], predictor=None):

        if isinstance(centroids, list):
            self.centroids = np.array(centroids)
        else:
            self.centroids = centroids

        self.predictor = predictor
        self.num_features = self.centroids.shape[1]
        self.num_clusters = self.centroids.shape[0]

    def explain(self,
                X: Union[np.ndarray, List],
                predictor=None) -> shap.Explanation:
        
        model = predictor if predictor is not None else self.predictor
        if model is None:
            raise ValueError("No predictor provided. Pass predictor argument or set self.predictor")


        X = np.atleast_2d(X)

        if hasattr(model, 'predict_proba'):
            model_fn = lambda x: model.predict_proba(x)
        elif callable(model):
            model_fn = model
        else:
            model_fn = lambda x: model.predict(x)

        explainer = shap.Explainer(model_fn, self.centroids)

        shap_values = explainer(X)

        return shap_values

    def get_centroids(self) -> np.ndarray:
        return self.centroids

    def set_predictor(self, predictor):
        self.predictor = predictor

    def __repr__(self):
        return (f"FedShapModel(n_clusters={self.num_clusters}, "
                f"n_features={self.num_features}, "
                f"has_predictor={self.predictor is not None})")