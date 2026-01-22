# run_generator.py (updated snippet)
import hydra_zen
import os
import pathlib
import hydra
from collections import OrderedDict
from hydra.utils import instantiate
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
import torch
import urllib.parse

def ordered_param_limit_dict(cfg):
    container = OmegaConf.to_container(cfg, resolve=True)
    return OrderedDict(container) if isinstance(container, dict) else OrderedDict()

try:
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))
except ValueError:
    # already registered
    pass

def _resolve_rml_path(cfg):
    # Get file_root and rml_basefile from cfg (user.yaml)
    #print("./rml/METRIX_U41_G1_H1_318eV_PS_MLearn_1.15.rml exist?!", os.path.exists("./rml/METRIX_U41_G1_H1_318eV_PS_MLearn_1.15.rml"))
    file_root = get_original_cwd()#cfg.get("file_root", "") or ""
    rml_basefile_cfg = cfg.get("rml_basefile", "")

    # If rml_basefile is expressed relative to file_root, expand it:
    if file_root:
        candidate = os.path.join(file_root, rml_basefile_cfg.lstrip("/"))
    else:
        candidate = rml_basefile_cfg

    # Expand user and make absolute
    candidate = os.path.expanduser(candidate)
    candidate_abs = os.path.abspath(candidate)

    # If the path exists on disk, convert to file:// URL (some backends expect URL)
    if os.path.exists(candidate_abs):
        file_url = pathlib.Path(candidate_abs).as_uri()   # yields file:///...
        return candidate_abs, file_url
    else:
        # If it doesn't exist, still try to interpret as a file:// URL (maybe user supplied one)
        parsed = urllib.parse.urlparse(candidate_abs)
        if parsed.scheme:
            # has a scheme (maybe file:// already) — return as-is
            return candidate_abs, candidate_abs
        # not found and no scheme -> raise helpful error
        raise FileNotFoundError(
            f"rml_basefile not found at resolved path: {candidate_abs}\n"
            "Set `file_root` in conf/user.yaml or set `rml_basefile` to a valid absolute path or file:// URL."
        )

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    #print("Composed config:")
    #print(OmegaConf.to_yaml(cfg))

    # instantiate backend + transform
    backend = instantiate(cfg.backend_options[cfg.selected_backend])
    transform = instantiate(cfg.transform_options[cfg.transform_selected])

    # Resolve rml_basefile (path + file:// URL)
    try:
        rml_path, rml_file_url = _resolve_rml_path(cfg)
    except FileNotFoundError as e:
        raise RuntimeError(str(e))

    #print("Using rml_basefile (path):", rml_path)
    #print("Using rml_basefile (file URL):", rml_file_url)

    # Instantiate engine while overriding rml_basefile to the absolute path or file URL
    # Try to pass the plain path first (engine might accept it). If that fails, pass file:// URL.
    try:
        engine = instantiate(cfg.engine, ray_backend=backend, rml_basefile=rml_path)
    except Exception as e_path:
        # try with file:// URL as fallback
        try:
            engine = instantiate(cfg.engine, ray_backend=backend, rml_basefile=rml_file_url)
        except Exception as e_url:
            # both attempts failed; show both trace info for diagnosis
            raise RuntimeError(
                "Failed to instantiate engine with rml_basefile as path and as file:// URL.\n"
                f"Error with path ({rml_path}): {e_path}\n"
                f"Error with file URL ({rml_file_url}): {e_url}"
            )

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
    print("Done.")

if __name__ == "__main__":
    main()

