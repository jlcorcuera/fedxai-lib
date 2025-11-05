import os
import pickle

import numpy as np
from fedlangpy.core.entities import FedlangEntity, pickle_io

from fedxai_lib.algorithms.federated_frbc.frbcs_fedxai_lib import FRBC_no_opt
from fedxai_lib.algorithms.federated_frt.utils.custom_logger import logger_info


class FederatedFRBCServer(FedlangEntity):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self, input_params=None):
        self.max_number_rounds = self.parameters.get("max_number_rounds")
        self.num_fuzzy_sets = self.parameters.get("num_fuzzy_sets")
        self.feature_number = self.parameters.get("num_features")
        self.model_output_file = self.parameters.get("model_output_file")
        self.feature_names = self.parameters.get("feature_names")
        self.unique_labels = self.parameters.get("unique_labels")

        
    
    @pickle_io
    def aggregate_local_RBs(self, local_RBs):
        """
        Aggregates the local rule bases from clients.
        :param local_RBs: List of local rule bases from clients.
        :return: Aggregated rule base.
        structure of local_RBs is a dict with keys:
        - 'antecedents': List of antecedents from each client.
        - 'consequents': List of consequents from each client.
        - 'certaintyFactors': List of certainty factors from each client.
        - 'CF_denominator': List of CF denominators from each client.
        """
        for local_RB in local_RBs:
            if not isinstance(local_RB, dict):
                raise ValueError("Local rule base must be a dictionary.")

            # Check for required keys
            required_keys = ['antecedents', 'consequents', 'certaintyFactors', 'CF_denominator']
            for key in required_keys:
                if key not in local_RB:
                    raise ValueError(f"Local rule base is missing required key: {key}")
        
        ## first aggregate all the local rule bases into a single RB with duplicates and conflicts
        first_client = local_RBs[0]

        antecedents = first_client['antecedents']
        consequents = first_client['consequents']
        certaintyFactors = first_client['certaintyFactors']
        cFs_denominators = first_client['CF_denominator']

        for col in local_RBs[1:]:
            antecedent = col['antecedents']
            consequent = col['consequents']
            certaintyFactor = col['certaintyFactors']
            cFs_denominator = col['CF_denominator']

            antecedents = np.concatenate((antecedents, antecedent), axis=0)
            consequents = np.concatenate((consequents, consequent), axis=0)
            certaintyFactors = np.concatenate((certaintyFactors, certaintyFactor), axis=0)
            cFs_denominators = np.concatenate((cFs_denominators, cFs_denominator), axis=0)

        ## filter duplicates and sum their contributions, then compute weights
        gb_antecedents, gb_consequents, gb_weights = self._sum_same_conseq_cfs_and_compute_rules_weight(
            antecedents, consequents, certaintyFactors, cFs_denominators)
        
        ## resolve conflicts
        gb_antecedents, gb_consequents, gb_weights = self._resolve_conflicts(gb_antecedents, gb_consequents, gb_weights)
        
        global_RB = {
            'antecedents': gb_antecedents,
            'consequents': gb_consequents,
            'rule_weights': gb_weights
        }
        self.global_RB = global_RB
        return global_RB
        

    def _sum_same_conseq_cfs_and_compute_rules_weight(self, antecedents, consequents, certaintyFactors, cFs_denominators):#, unique_labels):
    

        antecedents = np.concatenate((antecedents, consequents.reshape(-1,1)),axis=1)
        unq, count = np.unique(antecedents, axis=0, return_counts=True, return_index=False)
        repeated_groups = unq[count > 1]

        for repeated_group in repeated_groups:
            # Contain the indexes where are present the duplicated antecedent

            repeated_idx = np.argwhere(np.all(antecedents == repeated_group, axis = 1)).ravel()
            certaintyFactors[repeated_idx] = np.sum(certaintyFactors[repeated_idx])
            cFs_denominators[repeated_idx] = np.sum(cFs_denominators[repeated_idx])
        

        antecedents = np.concatenate((antecedents, certaintyFactors.reshape(-1,1), cFs_denominators.reshape(-1,1)), axis = 1)
        antecedents= np.unique(np.array(antecedents), axis=0)

        cFs_denominators = antecedents[:, -1]
        certaintyFactors = antecedents[:,-2]
        consequents = antecedents[:, -3]
        antecedents = antecedents[:, :-3]

        return antecedents, consequents, certaintyFactors/cFs_denominators
    
    def _resolve_conflicts(self, antecedents, consequents, rules_weight):
        unq, unique_indexes, count = np.unique(ar=antecedents, axis=0, return_counts=True,
                                               return_index=True)
        repeated_groups = unq[count > 1]

        for repeated_group in repeated_groups:
            # Contain the indexes where are present the duplicated antecedent
            repeated_idx = np.argwhere(np.all(antecedents == repeated_group, axis = 1)).ravel()
            # print("----- repeated idx -------")
            # print(repeated_idx)
            max_idx = np.argmax(rules_weight[repeated_idx])
            # print("max_idx : "+str(max_idx))
            # print(rules_weight[repeated_idx[0]])
            # print(rules_weight[repeated_idx[1]])
            
            # the idea is to change the records to match the "winner", then unique to remove duplicates
            consequents[repeated_idx] = consequents[repeated_idx[max_idx]]
            rules_weight[repeated_idx] = rules_weight[repeated_idx[max_idx]]
            
        # ******** RETRIEVE THE UNIQUE RULES **********#
        unique_antecedents = antecedents[unique_indexes]
        unique_rules_weight = rules_weight[unique_indexes]
        unique_consequents = consequents[unique_indexes]
        
        return unique_antecedents, np.asarray(unique_consequents).reshape(-1,1), unique_rules_weight


    ## this must be adapted with the new class model
    def build_model(self):
        dict_info ={
            'num_fs': self.num_fuzzy_sets, 
            'num_features': self.feature_number,
            'unique_labels': self.unique_labels,
            'feature_names': self.feature_names
        }
        frbc = FRBC_no_opt(dict_info)
        frbc._antecedents = self.global_RB['antecedents']
        frbc._consequents = self.global_RB['consequents']
        frbc.certaintyFactors = self.global_RB['rule_weights']
        frbc.weights = self.global_RB['rule_weights']
        self.model = frbc
        return
    
    def save_model_csv(self, input_params=None):
        
        if not hasattr(self, 'global_RB'):
            raise ValueError("No global rule base available to save the model.")
        if not hasattr(self, 'model'):
            self.build_model()
        
        path_folder = "./frbc_results"
        ##check if folder exists, if not create it
        
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        return self.model.save_model(path_folder)
    
    def save_model(self, input_params=None):
        if not hasattr(self, 'global_RB'):
            raise ValueError("No global rule base available to save the model.")
        if not hasattr(self, 'model'):
            self.build_model()
        output_file = self.model_output_file
        logger_info(f'model_output_file = {output_file}')
        with open(output_file, 'wb') as model_file:
            pickle.dump(self.model, model_file, pickle.HIGHEST_PROTOCOL)
        return self.model
    
FederatedFRBCServer(type="server")