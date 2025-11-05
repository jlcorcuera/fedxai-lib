import pickle
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from fedlangpy.core.entities import FedlangEntity, pickle_io
from numba import jit, njit, prange
from simpful import FuzzySet, Triangular_MF

from fedxai_lib.algorithms.federated_frt.utils.custom_logger import logger_info


@njit(parallel=True)
def compute_certainty_factor_numba(membership_matrix, antecedents, consequents, y_train, num_features):#, num_fuzzy_sets, certaintyFactors, penalized_CF, CF_denominator):

    num_antec = antecedents.shape[0]
    
    certaintyFactors = np.zeros(num_antec, dtype=np.float32)
    penalized_CF = np.zeros(num_antec, dtype=np.float32)
    CF_denominator = np.zeros(num_antec, dtype=np.float32)

    for i in prange(num_antec):
        current_conseq = consequents[i]
        certaintyFactor_numerator = 0.0
        penalizedCertaintyFactor_numerator = 0.0
        certaintyFactor_denominator = 0.0
        for j in range(membership_matrix.shape[0]):
            current_membershipDegree = 1.0
            for z in range(num_features):
                current_membershipDegree *= membership_matrix[j, z, antecedents[i, z]]
            if current_membershipDegree == 0.0:
                continue
            certaintyFactor_denominator += current_membershipDegree
            if y_train[j] == current_conseq:
                certaintyFactor_numerator += current_membershipDegree
            else:
                penalizedCertaintyFactor_numerator += current_membershipDegree
        certaintyFactors[i]=certaintyFactor_numerator
        penalized_CF[i]=penalizedCertaintyFactor_numerator
        CF_denominator[i]=certaintyFactor_denominator
    return certaintyFactors, penalized_CF, CF_denominator

class FederatedFRBCClient(FedlangEntity):

    TERMS = {
        3: ['LOW', 'MEDIUM', 'HIGH'],
        5: ['VERYLOW', 'LOW', 'MEDIUM', 'HIGH', 'VERYHIGH'],
        7: ['EXTREMELOW', 'VERYLOW', 'LOW', 'MEDIUM', 'HIGH', 'VERYHIGH', 'EXTREMEHIGH'],
        # se in futuro ne aggiungi altri, basta inserirli qui
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.X_train = kwargs.get("X_train")
        self.y_train = kwargs.get("y_train")

    def init_frbc(self, input_params=None):
        
        self.num_fuzzy_sets = self.parameters.get("num_fuzzy_sets")
        self.num_features = self.parameters.get("num_features")
        self.model_output_file = self.parameters.get("model_output_file")
        self.feature_names = self.parameters.get("feature_names")


        if self.X_train is None:
            df = pd.read_csv(self.parameters.get('dataset_X_train'))
            self.X_train = df[self.feature_names].to_numpy()
            self.y_train = df['Classe'].to_numpy()

        FS_structures, FS_parameters_numba, partitions = self._build_fuzzy_structures(
            counts=[self.num_fuzzy_sets]
        )
        self.FS_parameters_numba = FS_parameters_numba[self.num_fuzzy_sets]
        self.FS_structures = FS_structures[0]
        self.partitions = partitions





    def _build_fuzzy_structures(self, counts):
        """
        counts: lista di numeri dispari (es. [3,5,7])
        restituisce:
        - FS_structures: lista di dict con FuzzySet creati e mapping
        - FS_parameters_numba: dict[int → dict[a,b,c numpy arrays]]
        """
        FS_structures = []
        FS_parameters_numba = {}
        partitions = None  # Initialize partitions

        for n in counts:
            terms = FederatedFRBCClient.TERMS.get(n)
            if terms is None:
                raise ValueError(f"Nessuna lista di termini definita per n={n}")
            # passo tra i centri
            step = 1.0/(n-1)
            centers = np.linspace(0, 1, n)
            a = centers - step
            b = centers
            c = centers + step

            # salvo i parametri per numba
            FS_parameters_numba[n] = {
                'a': a.astype(np.float32),
                'b': b.astype(np.float32),
                'c': c.astype(np.float32)
            }

            # costruisco i FuzzySet
            FS_dict = {}
            mapping = []
            for term, ai, bi, ci in zip(terms, a, b, c):
                fs = FuzzySet(function=Triangular_MF(a=ai, b=bi, c=ci), term=term)
                key = f"FS_{term.lower()}"
                FS_dict[key] = fs
                mapping.append(fs)

            FS_dict['FS_mapping'] = mapping
            FS_structures.append(FS_dict)

            # partitions will be set to the last computed value
            partitions = np.tile(np.array([0.0] + list(centers) + [1.0]), (self.num_features, 1))
        return FS_structures, FS_parameters_numba, partitions

    def _find_fuzzy_set(self, value: float):
        
        values = []
        for fs_idx in range(self.num_fuzzy_sets):
            fs = self.FS_structures['FS_mapping'][fs_idx]
            values.append(fs.get_value(value))  # This line is commented out in the original
            # values.append(fs.get_value(value))
        # values = np.array([fs.get_value(value) for fs in self.FS_structures.values()])
        max_value = np.max(values)
        max_index = np.argmax(values)
        
        # need to check if membership degree is zero for second-highest fuzzy set
        index_second_fs = self.num_fuzzy_sets - 2 
        #indexes w.r.t. values, sorted by value in ascending order.  
        sort_index = [i for i,x in sorted(enumerate(values), key = lambda x:x[1])]
        
        if values[sort_index[index_second_fs]] == 0:
            return max_index, max_value-0.00001, sort_index[index_second_fs], 0.00001
        else:
            return max_index, max_value, sort_index[index_second_fs], values[sort_index[index_second_fs]]

    def _compute_firing_strengths(self, antecedents, input_vector: np.ndarray, t_norm: str = 'product') -> np.ndarray:
        """
        Compute the firing strength of each rule given a certain input

        :param antecedents: the rules antecedents with shape(num_rules, num_input_features)
        :param input_vector: vector in the form (num_input_features,)
        :param t_norm: the t-norm to be used in the computation of the firing strength of the antecedent
        :return: array with shape (num_rules,)
        """
        

        if t_norm != 'min' and t_norm != 'product':
            raise ValueError('invalid t-norm')

        list_firing_strengths = list()

        for rule in antecedents:
            activations_values = list()
            for elem, value in zip(rule, input_vector):
                fuzzy_set = self.FS_structures["FS_mapping"][int(elem)]
                membership_value = fuzzy_set.get_value(value)
                activations_values.append(membership_value)

            rule_firing_strength = 0
            if t_norm == 'min':
                rule_firing_strength = min(activations_values)
            if t_norm == 'product':
                rule_firing_strength = np.prod(activations_values)

            list_firing_strengths.append(rule_firing_strength)
        return np.array(list_firing_strengths)
    

    def _fit_parallelized(self, x_training, y_training):

        antecedents, membershipTable = self._generate_antecedents_data(x_training)
        self._antecedents, self._consequents = self._remove_duplicate_rules(antecedents=antecedents, consequents=self.y_train)
        

        self.certaintyFactors, self.penalized_CFs, self.CFs_denominator = compute_certainty_factor_numba(membershipTable, self._antecedents, self._consequents, y_training, self.num_features)
        self.weights = np.divide(self.certaintyFactors,self.CFs_denominator)
        return self._antecedents, self._consequents, self.certaintyFactors, self.CFs_denominator
    
    def _generate_antecedents_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        antecedents = list()
        num_features = self.num_features
        num_fs = self.num_fuzzy_sets
        mem_numpy = np.zeros([data.shape[0], num_features, num_fs])
        index_sample = 0
        for single_data_instance in data:
            antecedent = list()
            index_feature = 0
            
            # print("------- NEW DATA INSTANCE---------")
            for elem in single_data_instance:
                max_fs_index, max_fs_value, low_fs_index, low_fs_value  = self._find_fuzzy_set(value=elem)
                
                antecedent.append(max_fs_index)
                
                mem_numpy[index_sample, index_feature, max_fs_index] = max_fs_value
                mem_numpy[index_sample, index_feature, low_fs_index] = low_fs_value
                
               
                index_feature+=1
            index_sample +=1
            antecedents.append(antecedent)
        return np.array(antecedents), mem_numpy

    def _remove_duplicate_rules(self, antecedents: np.ndarray, consequents: np.ndarray):

        assert len(antecedents) == len(consequents), "Antecedents and consequents must have the same length"
    
        antecedents = np.concatenate((antecedents, consequents.reshape(-1, 1)), axis=1)
        antecedents= np.unique(np.array(antecedents), axis=0)
        
        consequents = antecedents[:, -1]
        antecedents = antecedents[:, :-1]
        
        return antecedents, consequents
    
    @pickle_io
    def generate_local_RB(self, input_params=None) -> dict:

        """
        Generates the local rule base from the training data.
        :return: Antecedents and consequents of the local rule base.
        """
        antecedents, consequents, certaintyFactors, CF_denominator = self._fit_parallelized(self.X_train, self.y_train)
        dict_rb = {
            'antecedents': antecedents,
            'consequents': consequents,
            'certaintyFactors': certaintyFactors,
            'CF_denominator': CF_denominator
        }
        return dict_rb
    
    def save_model(self, input_params=None):
        final_model = input_params
        logger_info(f'Receiving final model: {type(type(final_model))}')
        output_file = self.model_output_file
        model_client_id = output_file.replace(".pickle", f"_client_{self.id}.pickle")
        logger_info(f'model_output_file = {model_client_id}')
        with open(model_client_id, 'wb') as model_file:
            pickle.dump(final_model, model_file, pickle.HIGHEST_PROTOCOL)


FederatedFRBCClient(type="client")


