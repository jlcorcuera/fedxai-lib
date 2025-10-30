# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

import pickle
from fedlangpy.core.entities import pickle_io
from fedxai_lib.algorithms.federated_fcmeans_horizontal.server import FederatedHorizontalFCMServer
from fedxai_lib.algorithms.federated_shap.model import FedShapModel


class FederatedShapServer(FederatedHorizontalFCMServer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shap_model = None

    @pickle_io
    def build_shap_model(self, input_params=None):

        final_centroids = self.cluster_centers[-1]
        self.shap_model = FedShapModel(centroids=final_centroids)
        return self.shap_model

    @pickle_io
    def save_model(self, input_params=None):
        if self.shap_model is None:
            self.build_shap_model()

        model_path = self.parameters.get('model_save_path', 'fedshap_model.pkl')

        with open(model_path, 'wb') as f:
            pickle.dump(self.shap_model, f)

        return {
            'centroids': self.cluster_centers[-1]
        }

    def get_model(self) -> FedShapModel:

        if self.shap_model is None:
            self.build_shap_model()
        return self.shap_model


FederatedShapServer(type='server')
