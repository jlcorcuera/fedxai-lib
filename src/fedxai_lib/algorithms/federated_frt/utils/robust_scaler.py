# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

"""
Created on Jul. 20 09:37 a.m. 2024

@author: AI group, Department of Information Engineering, University of Pisa
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin


class RobustScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, columns, low_percentile, high_percentile):
        self.columns = columns
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

    def fit(self, X, y = None):
        self.params = {}
        for column in self.columns:
            percentile_h = X[column].quantile(self.high_percentile)
            percentile_l = X[column].quantile(self.low_percentile)
            self.params[column] = (percentile_l, percentile_h)
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for column in self.columns:
            percentile_l, percentile_h = self.params[column]
            if percentile_h - percentile_l == 0:
                X[column] = 0
            else:
                X[column] = (np.asarray(X[column]) - percentile_l) / (percentile_h - percentile_l)
            X.loc[X[column] > 1, column] = 1
            X.loc[X[column] < 0, column] = 0
        return X
