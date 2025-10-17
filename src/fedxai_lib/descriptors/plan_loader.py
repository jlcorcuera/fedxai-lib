import os
from enum import Enum
from fedlangpy.core.models import FLPlan
from fedlangpy.core.utils import load_plan


class PlanEnum(Enum):
    FED_FCMEANS_HORIZONTAL = "plan_federated_fcmeans_horizontal.json"
    FED_FCMEANS_VERTICAL = "plan_federated_fcmeans_vertical.json"
    FED_CMEANS_HORIZONTAL = "plan_federated_fcmeans_horizontal.json"
    FED_CMEANS_VERTICAL = "plan_federated_fcmeans_vertical.json"
    FED_FRT_HORIZONTAL = "plan_federated_frt.json"


def load_fedxai_plan(plan_enum: PlanEnum) -> FLPlan:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plan_location = os.path.join(current_dir, "definitions", plan_enum.value)
    return load_plan(plan_location)
