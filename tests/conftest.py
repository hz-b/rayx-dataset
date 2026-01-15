import numpy as np
import torch
import pytest

from ray_tools.simulation.data_tools import EfficientRandomRayDatasetGenerator
from ray_tools.base.engine import Engine

# --- DummyEngine used by integration tests ---
class DummyEngine(Engine):
    def __init__(self, variant:str="simple"):
        self.variant = variant

    def run(self, parameters, transforms=None):
        """
        parameters: torch.Tensor [B, P]
        transforms: list length B
        Return: list of dict results with key "ray_output".
        We keep shapes small and deterministic.
        """
        out = []
        B = parameters.shape[0]
        for i in range(B):
            if self.variant == "simple":
                # return a dict with scalar + 1D + 2D examples
                item = {
                    "ray_output": {
                        "plane1": {
                            "n_rays": 10,
                            "arr1d": np.arange(5),                  # 1D
                            "arr2d": np.arange(6).reshape(2, 3),    # 2D
                        }
                    }
                }
            else:
                # variant for other tests
                item = {
                    "ray_output": {
                        "plane1": {
                            "histogram": torch.ones((2, 4)),       # torch tensor 2x4
                            "meta": 42                              # scalar
                        }
                    }
                }
            out.append(item)
        return out

# ---- reusable sampler factory ----
def make_sampler(idx_sub=["a"], n_params=2):
    def sampler_func():
        # produce a single row of parameters (m==1)
        return torch.linspace(0.0, 1.0, n_params)
    return EfficientRandomRayDatasetGenerator.build_parameter_sampler(
        sampler_func=sampler_func, idx_sub=list(idx_sub), transform=None
    )

@pytest.fixture
def dummy_engine():
    return DummyEngine()

@pytest.fixture
def dummy_engine_variant():
    return DummyEngine(variant="hist")

@pytest.fixture
def sampler():
    return make_sampler(idx_sub=["s"], n_params=2)
