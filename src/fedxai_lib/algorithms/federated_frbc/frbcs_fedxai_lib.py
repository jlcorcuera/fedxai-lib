# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0


from collections import namedtuple

TensorKey = namedtuple('TensorKey', ['tensor_name', 'origin', 'round_number', 'report', 'tags'])

from typing import List, Tuple

import numpy as np
import pandas as pd
from numba import jit, njit, prange
from simpful import FuzzySet, Triangular_MF

# np.set_printoptions(formatter={'float': "{0:0.5f}".format})


class FRBC_no_opt():
    FS_structures = []

    FS_dict = {}
    FS_low = FuzzySet(function=Triangular_MF(a=-0.5, b=0, c=0.5), term='LOW')
    FS_medium = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term='MEDIUM')
    FS_high = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1.5), term='HIGH')

    FS_dict['FS_low'] = FS_low
    FS_dict['FS_medium'] = FS_medium
    FS_dict['FS_high'] = FS_high

    FS_mapping = [FS_low, FS_medium, FS_high]

    # ******** MAPPING ******** #
    # 0 -> FS_low
    # 1 -> FS_medium
    # 2 -> FS_HIGH
    # ******** MAPPING ******** #
    FS_dict['FS_mapping'] = FS_mapping
    FS_structures.append(FS_dict)

    # 5 fuzzy sets
    FS_dict = {}
    FS_verylow = FuzzySet(function=Triangular_MF(a=-0.25, b=0, c=0.25), term='VERYLOW')
    FS_low = FuzzySet(function=Triangular_MF(a=0, b=0.25, c=0.5), term='LOW')
    FS_medium = FuzzySet(function=Triangular_MF(a=0.25, b=0.5, c=0.75), term='MEDIUM')
    FS_high = FuzzySet(function=Triangular_MF(a=0.5, b=0.75, c=1), term='HIGH')
    FS_veryhigh = FuzzySet(function=Triangular_MF(a=0.75, b=1, c=1.25), term='VERYHIGH')

    FS_dict['FS_verylow'] = FS_verylow
    FS_dict['FS_low'] = FS_low
    FS_dict['FS_medium'] = FS_medium
    FS_dict['FS_high'] = FS_high
    FS_dict['FS_veryhigh'] = FS_veryhigh

    FS_mapping = [FS_verylow, FS_low, FS_medium, FS_high, FS_veryhigh]
    # ******** MAPPING ******** #
    # 0 -> FS_verylow
    # 1 -> FS_low
    # 2 -> FS_medium
    # 3 -> FS_high
    # 4 -> FS_veryhigh
    # ******** MAPPING ******** #
    FS_dict['FS_mapping'] = FS_mapping
    FS_structures.append(FS_dict)


    # 7 fuzzy sets
    FS_dict = {}
    FS_extremelow = FuzzySet(function=Triangular_MF(a=-0.1666, b=0, c=0.1666), term='EXTREMELOW')
    FS_verylow = FuzzySet(function=Triangular_MF(a=0, b=0.1666, c=0.3333), term='VERYLOW')
    FS_low = FuzzySet(function=Triangular_MF(a=0.1666, b=0.3333, c=0.5), term='LOW')
    FS_medium = FuzzySet(function=Triangular_MF(a=0.3333, b=0.5, c=0.6666), term='MEDIUM')
    FS_high = FuzzySet(function=Triangular_MF(a=0.5, b=0.6666, c=0.8333), term='HIGH')
    FS_veryhigh = FuzzySet(function=Triangular_MF(a=0.6666, b=0.8333, c=1), term='VERYHIGH')
    FS_extremehigh = FuzzySet(function=Triangular_MF(a=0.8333, b=1, c=1.1777), term='EXTREMEHIGH')

    FS_dict['FS_extremelow'] = FS_extremelow
    FS_dict['FS_verylow'] = FS_verylow
    FS_dict['FS_low'] = FS_low
    FS_dict['FS_medium'] = FS_medium
    FS_dict['FS_high'] = FS_high
    FS_dict['FS_veryhigh'] = FS_veryhigh
    FS_dict['FS_extremehigh'] = FS_extremehigh

    FS_mapping = [FS_extremelow, FS_verylow, FS_low, FS_medium, FS_high, FS_veryhigh, FS_extremehigh]
    # ******** MAPPING ******** #
    # 0 -> FS_extremelow
    # 1 -> FS_verylow
    # 2 -> FS_low
    # 3 -> FS_medium
    # 4 -> FS_high
    # 5 -> FS_veryhigh
    # 6 -> FS_extremehigh
    # ******** MAPPING ******** #
    FS_dict['FS_mapping'] = FS_mapping
    FS_structures.append(FS_dict)

    FS_parameters_numba = {
        3: {
            'a': np.array([-0.5, 0.0, 0.5], dtype=np.float32),
            'b': np.array([0.0, 0.5, 1.0], dtype=np.float32),
            'c': np.array([0.5, 1.0, 1.5], dtype=np.float32)
        },
        5: {
            'a': np.array([-0.25, 0.0, 0.25, 0.5, 0.75], dtype=np.float32),
            'b': np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32),
            'c': np.array([0.25, 0.5, 0.75, 1.0, 1.25], dtype=np.float32)
        },
        7: {
            'a': np.array([-0.1666, 0.0, 0.1666, 0.3333, 0.5, 0.6666, 0.8333], dtype=np.float32),
            'b': np.array([0.0, 0.1666, 0.3333, 0.5, 0.6666, 0.8333, 1.0], dtype=np.float32),
            'c': np.array([0.1666, 0.3333, 0.5, 0.6666, 0.8333, 1.0, 1.1777], dtype=np.float32)
        }
    }

    def __init__(self, dict_parameters):

        self.num_features = dict_parameters["num_features"]
        self.num_fuzzy_sets = dict_parameters["num_fs"]
        self.unique_labels = dict_parameters["unique_labels"]

        self.FS_parameters_numba = FRBC_no_opt.FS_parameters_numba[self.num_fuzzy_sets]
        if self.num_fuzzy_sets == 5:
            self.partitions = np.tile(np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]), (self.num_features, 1))
        elif self.num_fuzzy_sets == 3:
            self.partitions = np.tile(np.array([0.0, 0.0, 0.5, 1.0, 1.0]), (self.num_features, 1))
            
        
    @staticmethod
    def find_fuzzy_set(value: float, num_fs:int): #-> [int,float, int,float]:
        """
        Find the name of the fuzzy-set which is most activated
        :param value: the value to test
        :return: 
            the fuzzy-set mainly activated (using the mapping between a fuzzy set and an integer)
            the membership degree of each fuzzy set
        """
        
        num_fuzzy_set = num_fs
        
        if num_fuzzy_set == 3:
            membership_value_low = FRBC_no_opt.FS_structures[0]['FS_low'].get_value(value)
            membership_value_medium = FRBC_no_opt.FS_structures[0]['FS_medium'].get_value(value)
            membership_value_high = FRBC_no_opt.FS_structures[0]['FS_high'].get_value(value)

            values = [membership_value_low, 
                      membership_value_medium, 
                      membership_value_high]

            max_value = max(values)
            max_index = values.index(max_value)

        elif num_fuzzy_set == 5:
            membership_value_verylow = FRBC_no_opt.FS_structures[1]['FS_verylow'].get_value(value)
            membership_value_low = FRBC_no_opt.FS_structures[1]['FS_low'].get_value(value)
            membership_value_medium = FRBC_no_opt.FS_structures[1]['FS_medium'].get_value(value)
            membership_value_high = FRBC_no_opt.FS_structures[1]['FS_high'].get_value(value)
            membership_value_veryhigh = FRBC_no_opt.FS_structures[1]['FS_veryhigh'].get_value(value)

            values = [membership_value_verylow, 
                      membership_value_low, 
                      membership_value_medium, 
                      membership_value_high,
                      membership_value_veryhigh]

            max_value = max(values)
            max_index = values.index(max_value)

        elif num_fuzzy_set == 7:
            
            membership_value_extremelow = FRBC_no_opt.FS_structures[2]['FS_extremelow'].get_value(value)
            membership_value_verylow = FRBC_no_opt.FS_structures[2]['FS_verylow'].get_value(value)
            membership_value_low = FRBC_no_opt.FS_structures[2]['FS_low'].get_value(value)
            membership_value_medium = FRBC_no_opt.FS_structures[2]['FS_medium'].get_value(value)
            membership_value_high = FRBC_no_opt.FS_structures[2]['FS_high'].get_value(value)
            membership_value_veryhigh = FRBC_no_opt.FS_structures[2]['FS_veryhigh'].get_value(value)
            membership_value_extremehigh = FRBC_no_opt.FS_structures[2]['FS_extremehigh'].get_value(value)

            values = [membership_value_extremelow,
                      membership_value_verylow, 
                      membership_value_low, 
                      membership_value_medium, 
                      membership_value_high,
                      membership_value_veryhigh,
                      membership_value_extremehigh]
            
            max_value = max(values)
            max_index = values.index(max_value)

        
        index_second_fs = num_fuzzy_set - 2


        #indexes w.r.t. values, sorted by value.  
        sort_index = [i for i,x in sorted(enumerate(values), key = lambda x:x[1])]
        
        if values[sort_index[index_second_fs]] == 0:
            return max_index, max_value-0.00001, sort_index[index_second_fs], 0.00001
        else:
            return max_index, max_value, sort_index[index_second_fs], values[sort_index[index_second_fs]]


        
    def __get_rule_maximum_weight(self, weight_factor) -> Tuple[np.ndarray, int]:
        """
        Utility function used to retrieve the rule with the maximum weight
        :return: a tuple with:
                    - antecedent
                    - consequent
                    - index
        """
        if weight_factor == "CF":
            index_rule_weight = np.asarray(self.certaintyFactors).argmax()
        elif weight_factor == "PCF":
            index_rule_weight = self.penalizedCF.argmax()
        # print("-------------s")
        # print(self._antecedents[index_rule_weight,:])
        # print(self._consequents[index_rule_weight])
        # print(index_rule_weight)
        # print(self.certaintyFactors[index_rule_weight])
        return self._antecedents[index_rule_weight, :], self._consequents[index_rule_weight], index_rule_weight


    def __internal_predict(self, input_vector: np.ndarray, weight_factor:str):# -> Tuple[str, int]:
        
        rule_weight = 0
        firing_strengths = self.get_firing_strengths(antecedents=self._antecedents, input_vector=input_vector)

        is_all_zero = np.all((firing_strengths == 0))
        if is_all_zero:
            _, _, index_max_rules = self.__get_rule_maximum_weight(weight_factor="CF")
        else:
            max_values_firing = firing_strengths.max()
            index_same_firing_strengths = np.where(firing_strengths == max_values_firing)[0]
            
            if weight_factor == "CF":
                rule_weight = self.certaintyFactors[index_same_firing_strengths]
            elif weight_factor == "PCF":
                rule_weight = self.penalized_CFs[index_same_firing_strengths]
            index_max_rules = index_same_firing_strengths[rule_weight.argmax()]
            

        return self._consequents[index_max_rules], self.certaintyFactors[index_max_rules] 
    
    
    def predict_and_get_rule(self, X: np.ndarray, weight_factor: str):
        """
        Parameters
        ----------
        X : array-like  of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [2] or [n_samples, 2]
            the predicted values and activated rule.
        """
        return list(map(lambda input_vector: self.__internal_predict(input_vector, weight_factor=weight_factor), X))

    
            
    def save_model(self,path):

        np.savetxt(path+"/antecedents.csv", self._antecedents, delimiter=",")
        np.savetxt(path+"/consequents.csv", self._consequents, delimiter=",", fmt="%s")
        np.savetxt(path+"/certaintyFactor.csv", self.certaintyFactors, delimiter=",")
    

    def load_model(self, path):

        self._antecedents = np.loadtxt(path+"/antecedents.csv", delimiter=",")
        conseq = pd.read_csv(path+"/consequents.csv", header=None)
        self._consequents = conseq.values.reshape(-1,)
        self.certaintyFactors = np.loadtxt(path+"/certaintyFactor.csv", delimiter=",")
        self.weights = self.certaintyFactors
        
    
    
    def trl(self):
        return self._antecedents.shape[0]*self.num_features
        
    def get_rule(self, index_rule):
        antec = self._antecedents[index_rule]
        conseq = self._consequents[index_rule]

        return np.concatenate([antec, conseq])
    


    # #### NUMBA PARALLELIZATION
    
    ##### INFERENCE 

    @staticmethod
    @njit(parallel=True)
    def compute_firing_strengths_numba(membership_values, t_norm_num):
        """
        Calcola le firing strengths di ogni regola in parallelo utilizzando Numba.

        :param membership_values: Array 2D (num_rules, num_features) dei valori di membership.
        :param t_norm_num: 0 per 'min', 1 per 'product'.
        :return: Array 1D (num_rules,) delle firing strengths.
        """
        num_rules, num_features = membership_values.shape
        firing_strengths = np.empty(num_rules, dtype=np.float32)
        
        for i in prange(num_rules):
            if t_norm_num == 0:  # 'min' t-norm
                min_val = membership_values[i, 0]
                for j in range(1, num_features):
                    if membership_values[i, j] < min_val:
                        min_val = membership_values[i, j]
                firing_strengths[i] = min_val
            elif t_norm_num == 1:  # 'product' t-norm
                prod_val = 1.0
                for j in range(num_features):
                    prod_val *= membership_values[i, j]
                firing_strengths[i] = prod_val
            else:
                firing_strengths[i] = 0.0  # T-norm non supportato
        return firing_strengths

    
    
    def compute_firing_strengths_parallelized(self, antecedents, input_vector: np.ndarray, t_norm: str = 'product') -> np.ndarray:
        """
        Calcola le firing strengths di ogni regola dato un input.

        :param antecedents: Array delle antecedents delle regole con forma (num_rules, num_features).
        :param input_vector: Vettore di input normalizzato con forma (num_features,).
        :param t_norm: Tipo di t-norm da utilizzare ('min' o 'product').
        :return: Array delle firing strengths con forma (num_rules,).
        """
        # Determina l'indice dei fuzzy sets basato su num_fuzzy_sets
        if self.num_fuzzy_sets == 3:
            FS_idx = 0
        elif self.num_fuzzy_sets == 5:
            FS_idx = 1
        elif self.num_fuzzy_sets == 7:
            FS_idx = 2
        else:
            raise ValueError("Unsupported number of fuzzy sets")

        # Mappa il t_norm a un valore numerico
        if t_norm == 'min':
            t_norm_num = 0
        elif t_norm == 'product':
            t_norm_num = 1
        else:
            raise ValueError('Invalid t-norm')

        num_rules, num_features = antecedents.shape

        # Inizializza una matrice per memorizzare i valori di membership
        membership_values = np.empty((num_rules, num_features), dtype=np.float32)

        # Calcola i valori di membership utilizzando le funzioni FuzzySet esistenti
        for i in range(num_rules):
            for j in range(num_features):
                elem = antecedents[i, j]
                fuzzy_set = FRBC_no_opt.FS_structures[FS_idx]["FS_mapping"][int(elem)]
                membership_value = fuzzy_set.get_value(input_vector[j])
                membership_values[i, j] = membership_value

        # Chiama la funzione Numba per calcolare le firing strengths
        firing_strengths = FRBC_no_opt.compute_firing_strengths_numba(membership_values, t_norm_num)
        
        return firing_strengths

    def __internal_predict_parallelized(self, input_vector: np.ndarray, weight_factor:str):# -> Tuple[str, int]:
    
        rule_weight = 0
        firing_strengths = self.compute_firing_strengths_parallelized(antecedents=self._antecedents, input_vector=input_vector)

        is_all_zero = np.all((firing_strengths == 0))
        if is_all_zero:
            _, _, index_max_rules = self.__get_rule_maximum_weight(weight_factor="CF")
        else:
            max_values_firing = firing_strengths.max()
            index_same_firing_strengths = np.where(firing_strengths == max_values_firing)[0]
            if weight_factor == "CF":
                rule_weight = self.certaintyFactors[index_same_firing_strengths]
            elif weight_factor == "PCF": ## NOT IMPLEMENTED 
                rule_weight = self.penalized_CFs[index_same_firing_strengths]
            index_max_rules = index_same_firing_strengths[rule_weight.argmax()]
            
        return self._consequents[index_max_rules], index_max_rules
    
    def predict_and_get_rule_parallelized(self, X: np.ndarray, weight_factor: str):
        """
        Parameters
        ----------
        X : array-like  of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [2] or [n_samples, 2]
            the predicted values and activated rule.
        """
        return list(map(lambda input_vector: self.__internal_predict_parallelized(input_vector, weight_factor=weight_factor), X))


    
    
