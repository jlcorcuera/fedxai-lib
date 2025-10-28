# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0


from collections import namedtuple

TensorKey = namedtuple('TensorKey', ['tensor_name', 'origin', 'round_number', 'report', 'tags'])

import numpy as np
from simpful import FuzzySet, Triangular_MF
from typing import List, Tuple
import pandas as pd

from numba import njit, jit, prange

# np.set_printoptions(formatter={'float': "{0:0.5f}".format})


@jit(nopython=True)
def membership_degree(mf, x, fuzzy_t="triangular"):
    """Get membership degree between a fuzzy set and a crisp value

    Parameters
    ----------
    mf: list, shape()
        membership function described by its key-points
    x: float
        crisp value for which we want to compute the membership degree
    fuzzy_t: string, optional, default = 'triangular'
        shape of the membership function

    Returns
    -------
    degree: float
        membership degree of mf(x)

    """
    assert len(mf) >= 3, "A fuzzy set has at least 3 elements"
    assert mf[0] <= mf[1] <= mf[2], "membership function is wrong"
    assert mf[-1] - mf[0] > 0, "support cannot be zero"

    if np.isnan(x):
        return 1.0

    if fuzzy_t == "triangular" and len(mf) == 3:
        if mf[0] == mf[1]:  # left triangular
            if x < mf[1] == 0.0:
                return 1.0
            elif x > mf[2] or (x < mf[1] != 0.0):
                return 0.0
            else:
                return 1.0 - ((x - mf[1]) / (mf[2] - mf[1]))
        elif mf[1] == mf[2]:  # right triangular
            if x > mf[1] == 1.0:
                return 1.0
            elif x < mf[0] or (x > mf[1] != 1.0):
                return 0.0
            else:
                return (x - mf[0]) / (mf[1] - mf[0])
        else:  # triangular
            if x < mf[0] or x > mf[2]:
                return 0.0
            elif x <= mf[1]:
                return (x - mf[0]) / (mf[1] - mf[0])
            elif x <= mf[2]:
                return 1.0 - ((x - mf[1]) / (mf[2] - mf[1]))
    # Not implemented
    return 0.0


@jit(nopython=True)
def predict_fast(x, ant_matrix, cons_vect, weights, part_matrix):
    """Predict a class applying fuzzy rule-based inference

    Parameters
    ----------
    x: np.array, shape(n_sample, n_features)
        input data
    ant_matrix: np.array, shape(n_rules, n_features * n_fuzzy_sets)
        antecedents of every rule in the RB
    cons_vect: np.array, shape(n_rules, 1)
        consequents of every rule corresponding to one possible class
    weights: np.array, shape(n_rules, 1)
        weights for each rule in the RB
    part_matrix: np.array, shape(?, ?)

    Returns
    -------
    y: np.array, shape(n_sample,)
        predicted output for the input data x

    Notes
    -----
        The prediction uses the maximum matching method as reasoning method:
        an input is classified into the class corresponding
        to the rule with the maximum association degree
    """
    sample_size = x.shape[0]
    y = np.zeros(sample_size)
    # For each sample
    for i in range(sample_size):
        best_index = 0
        best_association_degree = 0.0
        # For each rule
        for j in range(ant_matrix.shape[0]):
            matching_degree = 1.0
            for k in range(ant_matrix.shape[1]):
                if ant_matrix[j][k] != -1:
                    base = int(ant_matrix[j][k])
                    ant = part_matrix[k][base: base + 3]
                    m_degree = membership_degree(ant, x[i][k])
                    matching_degree *= m_degree
            association_degree = weights[j] * matching_degree
            if association_degree > best_association_degree:
                best_index = j
                best_association_degree = association_degree
        y[i] = cons_vect[best_index]
    return y


@njit(parallel=True)
def predict_fast_parallel(x, ant_matrix, cons_vect, weights, part_matrix):
    """
    Versione parallelizzata della funzione predict_fast.
    Si sfrutta prange sul loop esterno (campioni) e si in-linea il calcolo del grado di appartenenza.
    
    Parametri:
      - x: array dei campioni (n_samples, n_features)
      - ant_matrix: array degli antecedenti (n_rules, n_features)
          (si assume che un valore -1 indichi che per quella feature la regola non è attiva)
      - cons_vect: array dei consequents (n_rules,)
      - weights: array dei pesi per ogni regola (n_rules,)
      - part_matrix: array delle partizioni per ciascuna feature (n_features, num_partizioni)
          dove per ogni feature i, per una regola che usa il fuzzy set con indice base,
          si prendono i 3 valori: part_matrix[i, base], part_matrix[i, base+1], part_matrix[i, base+2]
    
    Restituisce:
      - y: array dei predetti (n_samples,)
    """
    sample_size = x.shape[0]
    num_rules = ant_matrix.shape[0]
    num_features = ant_matrix.shape[1]
    y = np.empty(sample_size, dtype=np.float32)
    
    for i in prange(sample_size):
        best_index = 0
        best_association_degree = 0.0
        # Per ogni regola
        for j in range(num_rules):
            matching_degree = 1.0
            # Per ogni feature
            for k in range(num_features):
                # Se la regola prevede un fuzzy set per la feature k
                if ant_matrix[j, k] != -1:
                    base = int(ant_matrix[j, k])
                    # Recupera i parametri del fuzzy set (triangolare)
                    a = part_matrix[k, base]
                    b = part_matrix[k, base + 1]
                    c = part_matrix[k, base + 2]
                    x_val = x[i, k]
                    # Calcolo inline del grado di appartenenza
                    if x_val < a or x_val > c:
                        m_degree = 0.0
                    elif x_val <= b:
                        # Evita divisione per zero
                        if b - a == 0:
                            m_degree = 1.0
                        else:
                            m_degree = (x_val - a) / (b - a)
                    else:
                        if c - b == 0:
                            m_degree = 1.0
                        else:
                            m_degree = (c - x_val) / (c - b)
                    matching_degree *= m_degree
            association_degree = weights[j] * matching_degree
            if association_degree > best_association_degree:
                best_association_degree = association_degree
                best_index = j
        y[i] = cons_vect[best_index]
    return y


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
        # print("------- value to be analyzed -------")
        # print(value)
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
        # print("-------- values -----------")
        # print(values)
        # print("---------indexes ---------")
        # print(sort_index)
        if values[sort_index[index_second_fs]] == 0:
            
            # print("--------------------------------")
            # print(max_index, max_value-0.00001, sort_index[index_second_fs], 0.00001)
            return max_index, max_value-0.00001, sort_index[index_second_fs], 0.00001
        else:

            # print(max_index, max_value, sort_index[index_second_fs], values[sort_index[index_second_fs]])
            return max_index, max_value, sort_index[index_second_fs], values[sort_index[index_second_fs]]

    #@staticmethod
    def compute_firing_strengths(self, antecedents, input_vector: np.ndarray, t_norm: str = 'product') -> np.ndarray:
        """
        Compute the firing strength of each rule given a certain input

        :param antecedents: the rules antecedents with shape(num_rules, num_input_features)
        :param input_vector: vector in the form (num_input_features,)
        :param t_norm: the t-norm to be used in the computation of the firing strength of the antecedent
        :return: array with shape (num_rules,)
        """
        if self.num_fuzzy_sets == 3:
            FS_idx = 0
        elif self.num_fuzzy_sets == 5:
            FS_idx = 1

        if t_norm != 'min' and t_norm != 'product':
            raise ValueError('invalid t-norm')

        list_firing_strengths = list()

        for rule in antecedents:
            activations_values = list()
            for elem, value in zip(rule, input_vector):
                fuzzy_set = FRBC_no_opt.FS_structures[FS_idx]["FS_mapping"][int(elem)]
                membership_value = fuzzy_set.get_value(value)
                activations_values.append(membership_value)

            rule_firing_strength = 0
            if t_norm == 'min':
                rule_firing_strength = min(activations_values)
            if t_norm == 'product':
                rule_firing_strength = np.prod(activations_values)

            list_firing_strengths.append(rule_firing_strength)
        return np.array(list_firing_strengths)

    def get_firing_strengths(self, antecedents : np.ndarray, input_vector: np.ndarray) -> np.ndarray:
        """
        Compute the firing strength of each rule given a certain input
        :param input_vector: vector in the form (number_of_input_features,)
        :return: array with shape (num_rules,)
        """

        firing_strengths = self.compute_firing_strengths(antecedents=antecedents,
                                                                input_vector=input_vector)

        return firing_strengths





    def _remove_duplicate_rules(self, antecedents, consequents):
        print(" ------len antec & conseq ------")
        print(len(antecedents))
        print(len(consequents))
        if len(antecedents) != len(consequents):
            print("error, different dimensions")
            return 
        

    
        print(" ------unique labels ------")
        print(self.unique_labels)

        for i in range(0, len(antecedents)):
            # print("consequent i ")
            # print(consequents[i])
            antecedents[i].append(self.unique_labels.index(consequents[i]))

        #to add directly str to antec, then do unique and then remove last column, stop.
        # np.concatenate((antecedents,consequents), axis=1)
        antecedents= np.unique(np.array(antecedents), axis=0)
        
        
        consequents = antecedents[:, -1]
        antecedents = antecedents[:, :-1]
        print("---------consequents pre-------------")
        print(consequents)
        # consequents = [self.unique_labels[consequents[i]] for i in range(0, len(consequents))]
        # print("---------consequents---------")
        # print(consequents)
        
        return np.asarray(antecedents), np.asarray(consequents)


    
    def _compute_certaintyFactor(self, membership_matrix, consequents, antecedents, data, data_label):
        '''
            membership_matrix is a numpy ndarray of dimension : [N_sample, M_features, Z_fuzzy_sets] 
            with 
                N_sample        number of samples in training set
                M_features      number of features in the training set, per sample
                Z_fuzzy_sets    number of fuzzy sets employed

            For each rule, we compute certaintyFactor numerator, penalizedCertaintyFactor numerator,
            denominator to compute both certaintyFactor and penalizedCertaintyFactor    
        '''
        ## visualizza in colonne della membership per ottenere i vettori dei FS e poi moltiplichi quelli. Infine paragoni vettore mmebership con vettore classi per matching?
        # print(consequents)
        print("membership table shape "+ str(membership_matrix[1].shape))
        print("dimension input: " + str(data.shape))
        print("try access antec : "+str(antecedents[1][1]))
        num_features = data.shape[1]
        num_fuzzy_sets = 3
        certaintyFactors = []
        penalized_CF = []
        CF_denominator = []

        # flog = open("./conseq_log.csv", "w")
        # flog.write("antec_index,sample_index,antec_conseq,sample_label\n")
        
        #for each rule we compute the values for the certaintyFactor and penalized certaintyFactor
        for i in range(0, len(antecedents)):
            current_conseq = consequents[i]
            
            counter_match_conseq = 0
            #sum of matching degrees of all patterns in the fuzzy region for the specific class
            certaintyFactor_numerator = 0
            #sum of matching degrees of all patterns in the fuzzy region for samples without the specific class 
            penalizedCertaintyFactor_numerator = 0
            #sum of matching degrees of all patterns in the fuzzy region
            certaintyFactor_denominator = 0
            for j in range(0, len(data)):
                '''
                j index of data sample
                i num feature
                antecedents[i] = value of the fuzzy set for that feature
                '''
                # flog.write(str(i) + ","+str(j)+","+current_conseq+","+data_label[j]+"\n")
                current_membershipDegree = 1
                current_membershipDegree = current_membershipDegree *np.prod([membership_matrix[j, z, antecedents[i][z]] for z in range(0, num_features)])
                
                #if the sample doesn't activate the rule, continue to next sample
                if current_membershipDegree == 0:
                    continue
                # print("current membmership degree" + str(current_membershipDegree))

                if data_label[j] == self.unique_labels[current_conseq]:
                    counter_match_conseq +=1
                    certaintyFactor_numerator += current_membershipDegree
                else:
                    penalizedCertaintyFactor_numerator += current_membershipDegree
                certaintyFactor_denominator += current_membershipDegree
            certaintyFactors.append(certaintyFactor_numerator)
            penalized_CF.append(penalizedCertaintyFactor_numerator)
            CF_denominator.append(certaintyFactor_denominator)
        # flog.close()
        return np.asarray(certaintyFactors), np.asarray(penalized_CF), np.asarray(CF_denominator)
    
    def resolve_conflicting_rules(self):
        unq_antec,unq_indexes,count = np.unique(ar=self._antecedents, axis=0, return_counts=True, return_index=True)
        print("len antec: "+str(self._antecedents.shape))
        print("len conseq: "+ str(self._consequents.shape))
        print(np.concatenate((self._antecedents, self._consequents.reshape(-1,1)), axis=1))
        
        repeated_antecedents = unq_antec[count > 1]

        for repeated_antec in repeated_antecedents:
            repeated_idx = np.argwhere(np.all(self._antecedents == repeated_antec, axis = 1)).ravel()
            conseq_same_antec = self._consequents[repeated_idx]
            rule_weights = (self.certaintyFactors/self.CFs_denominator)[repeated_idx]
            print("--------- repeated idx ----------")
            print(repeated_idx)
            print(repeated_antec)
            print(conseq_same_antec)
            print(rule_weights)

            print(repeated_idx[np.argmax(rule_weights)])
            self._consequents[repeated_idx[np.argmin(rule_weights)]] = self._consequents[repeated_idx[np.argmax(rule_weights)]]
            self.certaintyFactors[repeated_idx[np.argmin(rule_weights)]] = self.certaintyFactors[repeated_idx[np.argmax(rule_weights)]]
            self.CFs_denominator[repeated_idx[np.argmin(rule_weights)]] = self.CFs_denominator[repeated_idx[np.argmax(rule_weights)]]
            
        self._antecedents = self._antecedents[unq_indexes]
        self._consequents = self._consequents[unq_indexes]    
        self.certaintyFactors = self.certaintyFactors[unq_indexes]
        self.CFs_denominator = self.CFs_denominator[unq_indexes]
        self.print_model(mode="verbose")


        return

    def _generate_antecedents_data(self, data: np.ndarray, training_infos: dict) -> Tuple[np.ndarray,np.ndarray]:
        
        ## applica fuzzy_set direttamente su tutto (???)
        weights = list()
        antecedents = list()
        # mem_sample = []
        # mem_feature = []
        # mem_fs_index = []
        # mem_fs_values = []
        num_features = training_infos["num_features"]
        num_fs = training_infos["num_fuzzy_sets"]
        mem_numpy = np.zeros([data.shape[0], num_features, num_fs])
        index_sample = 0
        # membership_table = membershipDict(training_infos)
        # membership_table.print()
        for single_data_instance in data:
            antecedent = list()
            weight = 1
            index_feature = 0
            
            # print("------- NEW DATA INSTANCE---------")
            for elem in single_data_instance:
                max_fs_index, max_fs_value, low_fs_index, low_fs_value  = FRBC_no_opt.find_fuzzy_set(value=elem, num_fs=self.num_fuzzy_sets)
                # print(" maxfs_ " + str(max_fs_index) + " maxfs_ " + str(max_fs_value) +" min " + str(low_fs_index) +" low val " + str(low_fs_value) )
                
                antecedent.append(max_fs_index)
                weight = weight * max_fs_value
                
                mem_numpy[index_sample, index_feature, max_fs_index] = max_fs_value
                mem_numpy[index_sample, index_feature, low_fs_index] = low_fs_value
                
                # print(str(weight) + " -- " + str(max_fs_value) + "---index " + str(max_fs_index))
                # update_membershipTable(mem_sample=mem_sample, mem_feature=mem_feature, mem_fs_index=mem_fs_index, mem_fs_values=mem_fs_values, num_features=num_fs,  \
                #                         index_sample=index_sample, index_feature=index_feature, max_fs_index=max_fs_index, max_fs_value=max_fs_value,\
                #                               low_fs_index=low_fs_index, low_fs_value=low_fs_value)
                
                
                # if index_sample > 19980:
                #     print(" index sample: " + str(index_sample) + "index_feature: "+str(index_feature)) 
                index_feature+=1
            index_sample +=1
            # if index_sample > 10:
            #     return
            weights.append(weight)
            antecedents.append(antecedent)
        # print("len mem_features: " +str(len(mem_feature)))
        # print("len mem_samples: " + str(len(mem_sample)))
        print("numpy matrix shape: " + str(mem_numpy.shape))

        
        return antecedents, weights, mem_numpy


    def predict_and_get_rule(self, x):
        
        return
    
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
            # print("index_same_firing_strengths shape: " + str(index_same_firing_strengths.shape))
            # print(" vlaue: "+ str(index_same_firing_strengths))
            if weight_factor == "CF":
                rule_weight = self.certaintyFactors[index_same_firing_strengths]
            elif weight_factor == "PCF":
                rule_weight = self.penalized_CFs[index_same_firing_strengths]
            index_max_rules = index_same_firing_strengths[rule_weight.argmax()]
            

            # print("--------- consequent result predict -----------")
            # print("index max rule")
            #print(index_max_rules)
            # print(rule_weight)
            # print(consequent_result)
        # return self._antecedents[index_max_rules], self._consequents[index_max_rules], self.certaintyFactors[index_max_rules] 
        return self._consequents[index_max_rules], self.certaintyFactors[index_max_rules] 
    
    
    def fit(self, x_training, y_training):
        
        training_infos = {"num_features": self.num_features, "num_fuzzy_sets":self.num_fuzzy_sets }
        antecedents, weights, membershipTable = self._generate_antecedents_data(x_training, training_infos)

        self._antecedents, self._consequents = self._remove_duplicate_rules(antecedents=antecedents, consequents=y_training)

        print(self._antecedents.shape)
        print(self._consequents.shape)

        self.certaintyFactors, self.penalized_CFs, self.CFs_denominator = self._compute_certaintyFactor(membershipTable, self._consequents, self._antecedents, x_training, y_training)
        # print("----------CFs-------------")
        # print("CF:" + str(self.certaintyFactors.shape))
        # print("CF:" + str(self.penalized_CFs.shape))
        # print("CF:" + str(self.CFs_denominator.shape))
        # print(self.certaintyFactors)
        # print("---------------------------")
        # print(self.penalized_CFs)
        # print("---------------------------")
        # print(self.CFs_denominator)
        self.resolve_conflicting_rules()
        
        return self._antecedents, self._consequents, self.certaintyFactors, self.CFs_denominator

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

    def print_model(self, mode):

        print("antecedent shape: " + str(self._antecedents.shape))
        print("consequent shape: " + str(self._consequents.shape))
        print("weights shape: " + str(self.certaintyFactors.shape))
        if mode == "verbose":
            print("----------antec, conseq----------")
            print(np.concatenate((self._antecedents,self._consequents.reshape(-1,1)),axis=1))
            print("----------CFs--------------")
            print(self.certaintyFactors/self.CFs_denominator)
            
    def save_model(self,path):

        np.savetxt(path+"/antecedents.csv", self._antecedents, delimiter=",")
        np.savetxt(path+"/consequents.csv", self._consequents, delimiter=",", fmt="%s")
        np.savetxt(path+"/certaintyFactor.csv", self.certaintyFactors, delimiter=",")
    

    def load_model(self, path):

        self._antecedents = np.loadtxt(path+"/antecedents.csv", delimiter=",")
        conseq = pd.read_csv(path+"/consequents.csv", header=None)
        self._consequents = conseq.values.reshape(-1,)
        self.certaintyFactors = np.loadtxt(path+"/certaintyFactor.csv", delimiter=",")

    def load_model_npy(self, path):
        
        self._antecedents = np.load(path+"antecedents.npy")
        self._consequents = np.load(path+"consequents.npy")
        self.certaintyFactors = np.load(path+"weights.npy")
        

    def train(self, col_name, round_num, data_loader):


        x_training = data_loader.get_train_data()
        y_training = data_loader.get_train_labels()
        antecedents, consequents, certainfyFactor, CFs_denominator = self.fit(x_training=x_training, y_training=y_training)        
        
        Rules = namedtuple('Rules', ['name', 'value'])
        rules = []
        rules.append(Rules(name="antecedents", value=antecedents))
        rules.append(Rules(name="consequents", value=consequents))
        rules.append(Rules(name="certaintyFactors", value=certainfyFactor))
        rules.append(Rules(name="CFs_denominator", value=CFs_denominator))

        origin = col_name
        output_rules_dict = {
            TensorKey(
                rule_set_name, origin, round_num, True, ('rule',)
            ): rule_set_value
            for (rule_set_name, rule_set_value) in rules
        }

        # self.save_xai_model(antecedents, consequents, rules_weight)

        return output_rules_dict
    
    def save_xai_model(self, rules_antec, rules_conseq, rules_weights):
        """
            Utility to save the Aggregated XAI model on aggregator's local file system. 
            The aggregated XAI TSK model consists of 3 numpy arrays: 
                - aggregated rule antecedents 
                - aggregated rule consequents 
                - aggregated rule weights 
        """
        

    def trl(self):
        return self._antecedents.shape[0]*self.num_features
        
    


    # #### GPU PARALLELIZATION
    # @staticmethod
    # @njit(parallel=True)
    # def generate_antecedents_data_numba( data, num_features, num_fs, mem_numpy, antecedents, weights):
    #     for index_sample in prange(data.shape[0]):
    #         weight = 1.0
    #         for index_feature in range(num_features):
    #             elem = data[index_sample, index_feature].values
    #             print(f"type of elem: {type(elem)}, elem: {elem}")
    #             max_fs_index, max_fs_value, low_fs_index, low_fs_value = FRBC_no_opt.find_fuzzy_set(elem, num_fs)
    #             antecedents[index_sample, index_feature] = max_fs_index
    #             weights[index_sample] *= max_fs_value
    #             mem_numpy[index_sample, index_feature, max_fs_index] = max_fs_value
    #             mem_numpy[index_sample, index_feature, low_fs_index] = low_fs_value
    #     return antecedents, weights, mem_numpy
    
    # def _generate_antecedents_data_gpu(self, data: np.ndarray, training_infos: dict) -> tuple:
    #     num_features = training_infos["num_features"]
    #     num_fs = training_infos["num_fuzzy_sets"]
    #     mem_numpy = np.zeros((data.shape[0], num_features, num_fs), dtype=np.float32)
    #     antecedents = np.zeros((data.shape[0], num_features), dtype=np.int32)
    #     weights = np.ones(data.shape[0], dtype=np.float32)
    #     antecedents, weights, mem_numpy = FRBC_no_opt.generate_antecedents_data_numba(data, num_features, num_fs, mem_numpy, antecedents, weights)
    #     return antecedents, weights, mem_numpy

    def fit_gpu(self, x_training, y_training):
        
        training_infos = {"num_features": self.num_features, "num_fuzzy_sets":self.num_fuzzy_sets }
        antecedents, weights, membershipTable = self._generate_antecedents_data(x_training, training_infos)

        self._antecedents, self._consequents = self._remove_duplicate_rules(antecedents=antecedents, consequents=y_training)
        
        print(self._antecedents.shape)
        print(self._consequents.shape)

        self.certaintyFactors, self.penalized_CFs, self.CFs_denominator = self._compute_certaintyFactor_gpu(membershipTable, self._consequents, self._antecedents, x_training, y_training)
        # print("----------CFs-------------")
        # print("CF:" + str(self.certaintyFactors.shape))
        # print("CF:" + str(self.penalized_CFs.shape))
        # print("CF:" + str(self.CFs_denominator.shape))
        # print(self.certaintyFactors)
        # print("---------------------------")
        # print(self.penalized_CFs)
        # print("---------------------------")
        # print(self.CFs_denominator)
        self.weights = np.divide(self.certaintyFactors,self.CFs_denominator)
        return self._antecedents, self._consequents, self.certaintyFactors, self.CFs_denominator
    

    @staticmethod
    @njit(parallel=True)
    def compute_certainty_factor_numba(membership_matrix, consequents, antecedents, data, data_label, num_features):#, num_fuzzy_sets, certaintyFactors, penalized_CF, CF_denominator):
    
        certaintyFactors = np.zeros(len(antecedents), dtype=np.float32)
        penalized_CF = np.zeros(len(antecedents), dtype=np.float32)
        CF_denominator = np.zeros(len(antecedents), dtype=np.float32)
        
        for i in prange(antecedents.shape[0]):
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
                if data_label[j] == current_conseq:
                    certaintyFactor_numerator += current_membershipDegree
                else:
                    penalizedCertaintyFactor_numerator += current_membershipDegree
            certaintyFactors[i]=certaintyFactor_numerator
            penalized_CF[i]=penalizedCertaintyFactor_numerator
            CF_denominator[i]=certaintyFactor_denominator
        # flog.close()
        return certaintyFactors, penalized_CF, CF_denominator

    

    def _compute_certaintyFactor_gpu(self, membership_matrix, consequents, antecedents, data, data_label):
        num_features = data.shape[1]
        num_fuzzy_sets = membership_matrix.shape[2]
        self.certaintyFactors, self.penalized_CFs, self.CFs_denominator = FRBC_no_opt.compute_certainty_factor_numba(membership_matrix, consequents, antecedents, data, data_label, num_features)#, num_fuzzy_sets, certaintyFactors, penalized_CF, CF_denominator)
        return self.certaintyFactors, self.penalized_CFs, self.CFs_denominator
    

##### INFERENCE WITH GPUs

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

    
    
    def compute_firing_strengths_gpu(self, antecedents, input_vector: np.ndarray, t_norm: str = 'product') -> np.ndarray:
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

    def __internal_predict_gpu(self, input_vector: np.ndarray, weight_factor:str):# -> Tuple[str, int]:
    
        rule_weight = 0
        firing_strengths = self.compute_firing_strengths_gpu(antecedents=self._antecedents, input_vector=input_vector)

        is_all_zero = np.all((firing_strengths == 0))
        if is_all_zero:
            _, _, index_max_rules = self.__get_rule_maximum_weight(weight_factor="CF")
        else:
            max_values_firing = firing_strengths.max()
            index_same_firing_strengths = np.where(firing_strengths == max_values_firing)[0]
            # print("index_same_firing_strengths shape: " + str(index_same_firing_strengths.shape))
            # print(" vlaue: "+ str(index_same_firing_strengths))
            if weight_factor == "CF":
                rule_weight = self.certaintyFactors[index_same_firing_strengths]
            elif weight_factor == "PCF":
                rule_weight = self.penalized_CFs[index_same_firing_strengths]
            index_max_rules = index_same_firing_strengths[rule_weight.argmax()]
            

            # print("--------- consequent result predict -----------")
            # print("index max rule")
            #print(index_max_rules)
            # print(rule_weight)
            # print(consequent_result)
        # return self._antecedents[index_max_rules], self._consequents[index_max_rules], self.certaintyFactors[index_max_rules] 
        return self._consequents[index_max_rules], self.certaintyFactors[index_max_rules] 
    
    def predict_and_get_rule_gpu(self, X: np.ndarray, weight_factor: str):
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
        return list(map(lambda input_vector: self.__internal_predict_gpu(input_vector, weight_factor=weight_factor), X))

    def predict_gpu(self, X, partitions):
        
        y = predict_fast_parallel(X, self._antecedents, self._consequents, self.weights, self.partitions)
        return np.array(y).reshape(len(y))