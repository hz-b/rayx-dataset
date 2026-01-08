from __future__ import annotations

import abc
import torch
from collections.abc import Iterable 
from collections import OrderedDict

from joblib import Parallel, delayed

from .raypyng.rml import RMLFile
from .raypyng.xmltools import XmlElement

from . import RayTransformType
from .backend import RayBackend, RayOutput
from .transform import RayTransform
import threading

class Engine(abc.ABC):
    @abc.abstractmethod
    def run(self,
            parameters: torch.Tensor,
            transforms: RayTransformType | Iterable[RayTransformType] | None = None,
            ) -> list[dict]:
        pass
        
class RayEngine(Engine):
    """
    Creates an engine to run raytracing simulations.
    :param rml_basefile: RML-file to be used as beamline template.
    :param exported_planes: Image planes and component outputs to be exported.
    :param ray_backend: RayBackend object that actually runs the simulation.
    :param num_workers: Number of parallel workers for runs (multi-threading, NOT multi-processing).
        Use 1 for no single-threading.
    :param as_generator: If True, :func:`RayEngine.run` returns a generator so that runs are performs when iterating
        over it.
    :param verbose:
    """

    def __init__(self,
                 rml_basefile: str,
                 param_list: list[str],
                 exported_planes: list[str],
                 ray_backend: RayBackend,
                 num_workers: int = 1,
                 as_generator: bool = False,
                 verbose: bool = False,
                 manual_transform_xyz: tuple[str, str, str] | None = None,
                 manual_transform_plane: str | None = None,
                 ) -> None:
        super().__init__()
        self.rml_basefile = rml_basefile
        self.exported_planes = exported_planes
        self.ray_backend = ray_backend
        self.num_workers = num_workers
        self.as_generator = as_generator
        self.verbose = verbose
        self.param_list = param_list
        self.manual_transform_plane = manual_transform_plane
        self.manual_transform_xyz = manual_transform_xyz
        self._thread_local = threading.local()
        
        self.indices = None
        if self.manual_transform_xyz is not None:
            missing = [p for p in self.manual_transform_xyz if p not in self.param_list]

            if missing:
                raise ValueError(
                    f"Missing required parameters in param_list: {missing}"
                )

            self.indices = tuple(self.param_list.index(p) for p in self.manual_transform_xyz)

        # internal RMLFile object
        self._raypyng_rml = RMLFile(self.rml_basefile)
        self.template = self._raypyng_rml.beamline

    def _get_thread_rml(self):
        if not hasattr(self._thread_local, "rml"):
            self._thread_local.rml = RMLFile(self.rml_basefile)   # parse once per thread
            self._thread_local.template = self._thread_local.rml.beamline
        return self._thread_local.rml, self._thread_local.template
    
    def run(self,
            parameters: torch.Tensor,
            transforms: RayTransformType | Iterable[RayTransformType] | None = None,
            ) -> list[dict]:
        """
        Runs simulation for given (Iterable of) parameter containers.
        :param parameters: input parameters to be processed.
        :param transforms: :class:`ray_tools.base.transform.RayTransform` to be used.
            If a singleton, the same transform is applied everywhere.
            If Iterable of RayTransform (same length a parameters), individual transforms are applied to each
            parameter container. Transform can be also dicts of RayTransform specifying with transform to apply to
            which exported planes (keys must be same as ``RayEngine.exported_planes``).
        :return: (Iterable of) dict with ray outputs (field ``ray_output``,
            see also :class:`ray_tools.base.backend.RayBackend`) and used parameters for simulation
            (field ``param_container_dict``, dict with same keys as in ``parameters``).
        """

        # convert transforms into list if it was a singleton
        if transforms is None or isinstance(transforms, (RayTransform, dict)):
            transforms_list = parameters.shape[0] * [transforms]
        else:
            transforms_list = transforms
            
        # Iterable of arguments used for RayEngine._run_func
        _iter = ((run_params, transform) for (run_params, transform) in
                 zip(list(parameters), transforms_list))
        # multi-threading (if self.num_workers > 1)
        worker = Parallel(n_jobs=self.num_workers, verbose=self.verbose, backend='threading')
        jobs = (delayed(self._run_func)(*item) for item in _iter)
        result = worker(jobs)
        if not isinstance(result, list):
            raise Exception("The result must be a list if we input a list.")
        # extract only element if parameters was a singleton
        return result
    
    def _run_func(self,
                  parameters: torch.Tensor,
                  transform: RayTransformType | None = None,
                  ) -> dict:
        """
        This method performs the actual simulation run.
        """
        result = {'param_container': dict(), 'ray_output': None}

        # create a copy of RML template to avoid problems with multi-threading
        raypyng_rml_work, template_work = self._get_thread_rml()

        # write values in param_container to RML template and param_container
        for i, (key) in enumerate(self.param_list):
            if key not in self.manual_transform_xyz:
                value = parameters[i].item()
                element = self._key_to_element(key, template=template_work)
                element.cdata = str(value)
                result['param_container'][key] = value

        # call the backend to perform the run
        ray_output_all_planes = self.ray_backend.run(raypyng_rml=raypyng_rml_work,
                                                    exported_planes=self.exported_planes)
        for key, ray_output in ray_output_all_planes.items():
            if key == self.manual_transform_plane:
                # compute x and y direction for normalized z direction (zz_dir would be 1)
                xz_dir = ray_output.x_dir / ray_output.z_dir
                yz_dir = ray_output.y_dir / ray_output.z_dir
                idx = torch.tensor(self.indices, device=parameters.device)
                trans_x, trans_y, trans_z = parameters[idx]

                x_cur = ray_output.x_loc + xz_dir * trans_z + trans_x
                y_cur = ray_output.y_loc + yz_dir * trans_z + trans_y
                z_cur = ray_output.z_loc + trans_z

                ray_output_all_planes[key].x_loc = x_cur
                ray_output_all_planes[key].y_loc = y_cur
                ray_output_all_planes[key].z_loc = z_cur

        result['ray_output'] = ray_output_all_planes
        # apply transform (to each exported plane)
        if transform is not None:
            for plane in self.exported_planes:
                t = transform if isinstance(transform, RayTransform) else transform[plane]
                result['ray_output'][plane] = t(result['ray_output'][plane])
        return result

    def _key_to_element(self, key: str, template: XmlElement | None = None) -> XmlElement:
        """
        Helper function that retrieves an XML-subelement given a key (same format as in parameters).
        """
        if template is None:
            template = self.template
        component, param = key.split('.')
        if template is None:
            raise Exception("Template cannot be None.")
        element = template.__getattr__(component)
        if not isinstance(element, XmlElement):
            raise Exception("Element must be XmlElement.")
        return element.__getattr__(param)

class MinMaxRayEngine(RayEngine):
    def __init__(
        self,
        *args,
        param_limit_dict: OrderedDict[str, tuple[float, float]],
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        if not isinstance(param_limit_dict, OrderedDict):
            raise TypeError("param_limit_dict must be an OrderedDict")

        self.param_limit_dict = param_limit_dict

        # Pre-build tensors for fast vectorized denormalization
        mins = []
        maxs = []

        for name, (min_val, max_val) in param_limit_dict.items():
            if max_val <= min_val:
                raise ValueError(f"Invalid limits for '{name}': {min_val}, {max_val}")
            mins.append(min_val)
            maxs.append(max_val)

        # Shape: [n]
        self._mins = torch.tensor(mins, dtype=torch.float32)
        self._ranges = torch.tensor(maxs, dtype=torch.float32) - self._mins

    def denormalize(self, parameters: torch.Tensor) -> torch.Tensor:
        """
        parameters: Tensor of shape [batch, n], values in [0, 1]
        returns: Tensor of shape [batch, n] in physical parameter ranges
        """
        if parameters.ndim != 2:
            raise ValueError("parameters must have shape [batch, n]")

        if parameters.shape[1] != len(self.param_limit_dict):
            raise ValueError(
                f"Expected {len(self.param_limit_dict)} parameters, "
                f"got {parameters.shape[1]}"
            )

        if torch.any(parameters < 0) or torch.any(parameters > 1):
            raise ValueError("Normalized parameters must be in [0, 1]")

        # Move limits to same device
        mins = self._mins.to(parameters.device)
        ranges = self._ranges.to(parameters.device)

        # Vectorized denormalization:
        # physical = min + normalized * (max - min)
        return mins + parameters * ranges

    def run(
        self,
        parameters: torch.Tensor,
        transforms: RayTransformType | Iterable[RayTransformType] | None = None,
    ) -> list[dict]:

        denorm_params = self.denormalize(parameters)

        return super().run(
            denorm_params,
            transforms=transforms,
        )

class GaussEngine(Engine):
    """
    New input pattern:
      parameters: Tensor [batch, n] with columns matching self.param_list.
    Expected names in param_list:
      required: x_mean, y_mean, x_var, y_var, number_rays,
                direction_spread, x_dir, y_dir, z_dir
      optional: correlation_factor
    Output:
      list of dicts with keys: 'param_container' (dict) and 'ray_output' ({'ImagePlane': RayOutput})
    """

    def __init__(
        self,
        param_list: list[str],
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.param_list = param_list
        self.device = device

        required = {
            "x_mean", "y_mean", "x_var", "y_var", "number_rays",
            "direction_spread", "x_dir", "y_dir", "z_dir",
        }
        missing = [k for k in required if k not in self.param_list]
        if missing:
            raise ValueError(f"GaussEngine: missing required params in param_list: {missing}")

    def run(
        self,
        parameters: torch.Tensor,
        transforms: RayTransformType | list[RayTransformType] | None = None,
    ) -> list[dict]:

        if parameters.ndim != 2:
            raise ValueError("parameters must have shape [batch, n]")

        if parameters.shape[1] != len(self.param_list):
            raise ValueError(
                f"Expected {len(self.param_list)} parameters, got {parameters.shape[1]}"
            )

        # normalize transforms into list
        if transforms is None or isinstance(transforms, (RayTransform, dict)):
            transforms_list = parameters.shape[0] * [transforms]
        else:
            transforms_list = transforms

        if len(transforms_list) != parameters.shape[0]:
            raise ValueError("If transforms is a list, it must have length == batch size")

        dev = self.device if self.device is not None else parameters.device
        params_on_dev = parameters.to(dev)

        # indices for fast lookup
        idx = {name: i for i, name in enumerate(self.param_list)}
        has_corr = "correlation_factor" in idx

        outputs: list[dict] = []
        for b in range(params_on_dev.shape[0]):
            row = params_on_dev[b]

            x_mean = row[idx["x_mean"]].item()
            y_mean = row[idx["y_mean"]].item()
            x_var = row[idx["x_var"]].item()
            y_var = row[idx["y_var"]].item()
            n_rays = int(row[idx["number_rays"]].item())
            direction_spread = row[idx["direction_spread"]].item()

            x_dir0 = row[idx["x_dir"]].item()
            y_dir0 = row[idx["y_dir"]].item()
            z_dir0 = row[idx["z_dir"]].item()

            correlation_factor = row[idx["correlation_factor"]] if has_corr else torch.tensor(0.0, device=dev)

            # guardrails
            if n_rays <= 0:
                raise ValueError(f"number_rays must be > 0, got {n_rays}")
            if x_var < 0 or y_var < 0:
                raise ValueError(f"Variances must be >= 0, got x_var={x_var}, y_var={y_var}")

            mean0 = torch.tensor([0.0, 0.0], device=dev, dtype=torch.float32)
            cov = torch.diag(torch.tensor([x_var, y_var], device=dev, dtype=torch.float32))
            mvn = torch.distributions.MultivariateNormal(mean0, cov)

            samples = mvn.rsample(torch.Size([n_rays]))  # [n_rays, 2]

            # rotation by correlation_factor (angle)
            c = torch.cos(correlation_factor).to(dev)
            s = torch.sin(correlation_factor).to(dev)
            R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])  # [2,2]
            samples = samples @ R.T

            # shift by means
            samples = samples + torch.tensor([x_mean, y_mean], device=dev, dtype=torch.float32)

            # random direction spread
            samples_directions = torch.rand([n_rays, 3], device=dev) * float(direction_spread)

            ray_out = RayOutput(
                samples[:, 0],
                samples[:, 1],
                torch.zeros_like(samples[:, 0]),
                torch.tensor(x_dir0, device=dev).float() + samples_directions[:, 0],
                torch.tensor(y_dir0, device=dev).float() + samples_directions[:, 1],
                torch.tensor(z_dir0, device=dev).float() + samples_directions[:, 2],
                torch.ones_like(samples[:, 0]),
            )

            # apply transform if provided (singleton transform or per-sample item)
            t = transforms_list[b]
            if t is not None:
                # if dict, expect plane key 'ImagePlane'
                tt = t if isinstance(t, RayTransform) else t["ImagePlane"]
                ray_out = tt(ray_out)

            # build param_container dict (for traceability)
            param_dict = {name: row[i].item() for i, name in enumerate(self.param_list)}

            outputs.append({
                "param_container": param_dict,
                "ray_output": {"ImagePlane": ray_out},
            })

        return outputs


class SurrogateEngine(Engine):
    """
    New input pattern:
      parameters: Tensor [batch, n] with columns matching self.param_list

    This keeps the same VAE/non-VAE behavior, but feeds params as tensors.
    If your non-VAE model expects something else, see the fallback block.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        device: torch.device,
        is_vae: bool = True,
        param_list: list[str] | None = None,
        latent_size: int = 200,
        output_plane: str = "ImagePlane",
    ):
        super().__init__()
        self.is_vae = is_vae
        self.device = device
        self.model = model.to(device)
        self.param_list = param_list
        self.output_plane = output_plane

        self.latent_size = latent_size if is_vae else None

    def run(
        self,
        parameters: torch.Tensor,
        transforms: RayTransformType | Iterable[RayTransformType] | None = None,
    ) -> list[dict]:

        if parameters.ndim != 2:
            raise ValueError("parameters must have shape [batch, n]")

        if self.param_list is not None and parameters.shape[1] != len(self.param_list):
            raise ValueError(
                f"Expected {len(self.param_list)} parameters, got {parameters.shape[1]}"
            )

        # normalize transforms into list
        if transforms is None or isinstance(transforms, (RayTransform, dict)):
            transforms_list = parameters.shape[0] * [transforms]
        else:
            transforms_list = list(transforms)

        if len(transforms_list) != parameters.shape[0]:
            raise ValueError("If transforms is a list, it must have length == batch size")

        params_on_dev = parameters.to(self.device)

        outputs: list[dict] = []
        for b in range(params_on_dev.shape[0]):
            row = params_on_dev[b]                 # [n]
            params_tensor = row.unsqueeze(0)        # [1, n]

            # Run surrogate
            if self.is_vae and self.latent_size is not None:
                draw = torch.normal(
                    mean=torch.zeros(self.latent_size, device=self.device),
                    std=torch.ones(self.latent_size, device=self.device),
                ).view(1, -1)

                # expected: model.decode(z, params)
                model_out = self.model.decode(draw, params_tensor)
            else:
                # expected: model(params_tensor, ...)
                # If your non-VAE model needs a different signature, adjust here.
                try:
                    model_out = self.model(params_tensor)
                except TypeError:
                    # Fallback: pass a dict keyed by param_list if that’s what your model expects
                    if self.param_list is None:
                        raise
                    param_dict_for_model = {k: params_tensor[0, i].item() for i, k in enumerate(self.param_list)}
                    model_out = self.model(param_dict_for_model, 1, 1, 1)

            # Put output into the standard engine structure
            if self.param_list is None:
                param_dict = {f"p{i}": row[i].item() for i in range(row.numel())}
            else:
                param_dict = {name: row[i].item() for i, name in enumerate(self.param_list)}

            ray_output = {self.output_plane: model_out}

            # Apply transforms (if they are RayTransform, they must accept whatever model_out is)
            t = transforms_list[b]
            if t is not None:
                tt = t if isinstance(t, RayTransform) else t[self.output_plane]
                ray_output[self.output_plane] = tt(ray_output[self.output_plane])

            outputs.append({
                "param_container": param_dict,
                "ray_output": ray_output,
            })

        return outputs

