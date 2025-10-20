import tempfile
from importlib.resources import files
from fedlangpy.core.models import FLPlan
from fedlangpy.core.utils import load_plan


def load_fedxai_plan(json_descriptor: str) -> FLPlan:
    json_content = files("fedxai_lib.descriptors.definitions").joinpath(json_descriptor).read_text()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp.write(json_content)
        tmp_path = tmp.name

    try:
        return load_plan(tmp_path)
    finally:
        import os
        os.unlink(tmp_path)
