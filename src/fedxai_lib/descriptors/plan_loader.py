import tempfile
from importlib.resources import files
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
    json_content = files("fedxai_lib.descriptors.definitions").joinpath(plan_enum.value).read_text()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp.write(json_content)
        tmp_path = tmp.name

    try:
        return load_plan(tmp_path)
    finally:
        import os
        os.unlink(tmp_path)
