import hydra_zen # workaround for Hydra and Python 1.14 compatibility issue
import os
import hydra
from collections import OrderedDict
from hydra.utils import instantiate
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd
import torch

def ordered_param_limit_dict(cfg):
    container = OmegaConf.to_container(cfg, resolve=True)
    return OrderedDict(container) if isinstance(container, dict) else OrderedDict()

try:
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))
except ValueError:
    # already registered
    pass

def _resolve_rml_path(cfg):
    file_root = get_original_cwd()
    rml_basefile_cfg = cfg.get("rml_basefile", "")

    # If rml_basefile is expressed relative to file_root, expand it:
    if file_root:
        candidate = os.path.join(file_root, rml_basefile_cfg.lstrip("/"))
    else:
        candidate = rml_basefile_cfg

    # Expand user and make absolute
    candidate = os.path.expanduser(candidate)
    candidate_abs = os.path.abspath(candidate)

    return candidate_abs

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    # instantiate backend + transform
    backend = instantiate(cfg.backend_options[cfg.selected_backend])
    transform = instantiate(cfg.transform_options[cfg.transform_selected])

    # Resolve rml_basefile (path + file:// URL)
    try:
        rml_path = _resolve_rml_path(cfg)
    except FileNotFoundError as e:
        raise RuntimeError(str(e))

    # Instantiate engine while overriding rml_basefile to the absolute path or file URL
    # Try to pass the plain path first (engine might accept it). If that fails, pass file:// URL.
    engine = instantiate(cfg.engine, ray_backend=backend, rml_basefile=rml_path)
   
    # sampler runtime function
    def sampler(batch_len):
        return torch.rand(batch_len, len(cfg.param_limit_dict))

    # instantiate generator class directly (sampler is a python callable)
    from ray_tools.simulation.data_tools import EfficientRandomRayDatasetGenerator
    gen = EfficientRandomRayDatasetGenerator(
        engine=engine,
        sampler=sampler,
        transform=transform,
        h5_datadir=cfg.outputs_dir,
        param_limit_dict=cfg.param_limit_dict,
        exported_planes=list(cfg.exported_planes),
        h5_basename=cfg.dataset_generator.h5_basename,
        h5_max_size=cfg.dataset_generator.h5_max_size,
        fixed_output_size=cfg.dataset_generator.fixed_output_size,
    )

    gen.generate(h5_idx=cfg.generate.h5_idx, batch_size=cfg.generate.batch_size)

if __name__ == "__main__":
    main()

