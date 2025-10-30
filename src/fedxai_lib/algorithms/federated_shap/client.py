# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

import pickle
from fedlangpy.core.entities import pickle_io
from fedxai_lib.algorithms.federated_fcmeans_horizontal.client import FederatedHorizontalFCMClient
from fedxai_lib.algorithms.federated_shap.model import FedShapModel


class FederatedShapClient(FederatedHorizontalFCMClient):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shap_model = None

    @pickle_io
    def save_model(self, model_info):

        if isinstance(model_info, dict):
            centroids = model_info.get('centroids', [])
            self.cluster_centers = centroids
            self.shap_model = FedShapModel(centroids=centroids)
            if self.parameters.get('save_client_model', False):
                client_model_path = self.parameters.get('client_model_path', f'client_{self.id}_fshap_model.pkl')
                with open(client_model_path, 'wb') as f:
                    pickle.dump(self.shap_model, f)
        else:
            self.cluster_centers = model_info.get('centroids', [])

    def get_model(self) -> FedShapModel:
        if self.shap_model is None and hasattr(self, 'cluster_centers'):
            self.shap_model = FedShapModel(centroids=self.cluster_centers)
        return self.shap_model


FederatedShapClient(type='client')
