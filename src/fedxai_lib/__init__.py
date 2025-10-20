#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
  @Filename:    __init__.py
  @Author:      José Luis Corcuera Bárcena
  @Time:        7/4/25 11:45 AM
"""
from enum import Enum
from fedlangpy.core.utils import run_experiment
from fedxai_lib.descriptors.plan_loader import load_fedxai_plan


class FedXAIAlgorithm(Enum):
    def __init__(self, id, json_descriptor):
        self.id = id
        self.json_descriptor = json_descriptor

    FED_FCMEANS_HORIZONTAL = (1, "plan_federated_fcmeans_horizontal.json")
    FED_FCMEANS_VERTICAL = (2, "plan_federated_fcmeans_vertical.json")
    FED_CMEANS_HORIZONTAL = (3, "plan_federated_fcmeans_horizontal.json")
    FED_CMEANS_VERTICAL = (4, "plan_federated_fcmeans_vertical.json")
    FED_FRT_HORIZONTAL = (5, "plan_federated_frt.json")


def run_fedxai_experiment(algorithm: FedXAIAlgorithm, server, clients, parameters):
    fl_plan = load_fedxai_plan(algorithm.json_descriptor)
    run_experiment(fl_plan, server, clients, parameters)
