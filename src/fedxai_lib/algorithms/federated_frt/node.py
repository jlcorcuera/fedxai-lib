# Copyright (C) 2025 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

"""
Created on Jul. 20 09:37 a.m. 2024

@author: AI group, Department of Information Engineering, University of Pisa
"""
from typing import Final

NODE_ROOT_ID: Final[str] = "1"


class FedFRTFeatureFSStats:

    def __init__(self, all_zeros = False, **kwargs):
        if all_zeros:
            self.WSS = 0
            self.WLS = 0
            self.WS = 0
            self.S = 0
            self.NTH = 0
        else:
            self.WSS = kwargs.get("WSS")
            self.WLS = kwargs.get("WLS")
            self.WS = kwargs.get("WS")
            self.S = kwargs.get("S")
            self.NTH = kwargs.get("NTH")

class FedFRTNode:
    __LAST_ID = 0

    @staticmethod
    def reset_node_id():
        FedFRTNode.__LAST_ID = 0

    def __init__(self, **kwargs):
        FedFRTNode.__LAST_ID = FedFRTNode.__LAST_ID + 1
        self.id = FedFRTNode.__LAST_ID
        self.feature_name = kwargs.get("feature_name")
        self.feature = kwargs.get("feature", -1)
        self.fSet = kwargs.get("f_set")
        if self.fSet:
            self.membership_degree_fn = self.fSet.get_value
        self.depth = kwargs.get("depth", 0)
        self.parent = kwargs.get("parent")
        self.variance = kwargs.get("variance")
        self.gain = kwargs.get("gain")
        self.mean_activation_force = kwargs.get("mean_activation_force")
        self.num_samples = kwargs.get("num_samples")
        self.num_samples_above_thr = kwargs.get("num_samples_above_thr")
        self.is_leaf = kwargs.get("is_leaf", False)
        self.child = None

    def membership_degree(self, x):
        return self.membership_degree_fn(x[self.feature])

    def get_comp_id(self):
        parent_id = self.parent.id if self.parent else ''
        if parent_id:
            return f'{parent_id}_{self.feature}_{self.fSet.get_term()}'
        return str(self.id)

    def mark_as_leaf(self):
        self.is_leaf = True

    def add_children(self, node):
        if not self.child:
            self.child = list()
        self.child.append(node)

    def get_child(self):
        return self.child

    def _num_descendants(self):
        if self.is_leaf:
            return 1.
        else:
            return 1 + sum([child._num_descendants() for child in self.child])

    def _num_leaves(self):
        if self.is_leaf == 1:
            leaf = 1.
        else:
            leaf = 0.
        if self.child is not None:
            child_sum = map(lambda child: child._num_leaves(), self.child)
            for k in child_sum:
                leaf += k
        return leaf

    def __str__(self):
        if self.depth == 0:
            return f"feature: {self.feature_name}, f-set = {self.fSet}"
        return "\t" * self.depth + f'feature: {self.feature_name}, f-set = {self.fSet}'
