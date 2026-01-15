from __future__ import annotations

import abc
import torch
import numpy as np
from ray_tools.base.rml.transform import apply_rigid_transform
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
        Run simulation using parameters tensor and apply_rigid_transform to update
        worldPosition/world*direction in the XmlElement template (no apply_parameters call).
        """
        result = {'param_container': dict(), 'ray_output': None}

        # create a copy of RML template to avoid problems with multi-threading
        raypyng_rml_work, template_work = self._get_thread_rml()

        # --- Helpers for XmlElement usage ------------------------------------------------
        def find_beamline(root_elem):
            # Case 1: root itself IS the beamline
            if root_elem.name() == 'beamline':
                return root_elem

            # Case 2: beamline is a child
            b_elems = root_elem.get_elements('beamline')
            return b_elems[0] if b_elems else None

        def find_objects(beam_elem):
            # returns list of object XmlElement children
            return [o for o in beam_elem.get_elements('object')]

        def find_param(obj_elem, param_id):
            # find first param child whose attribute 'id' equals param_id
            for p in obj_elem.get_elements('param'):
                if p.get_attribute('id') == param_id:
                    return p
            return None

        def parse_vec_xml(param_elem):
            """
            Read a vector from:
              <param>
                <x>...</x>
                <y>...</y>
                <z>...</z>
              </param>
            """
            if param_elem is None:
                raise ValueError("param_elem is None")

            def get_val(axis):
                try:
                    node = getattr(param_elem, axis)
                    txt = node.cdata.strip()
                    return float(txt) if txt else 0.0
                except AttributeError:
                    # missing axis -> default 0
                    return 0.0

            x = get_val('x')
            y = get_val('y')
            z = get_val('z')
            return (x, y, z)

        def write_vec_xml(param_elem, vec):
            """
            Write a vector into:
              <param>
                <x>...</x>
                <y>...</y>
                <z>...</z>
              </param>
            """
            if param_elem is None:
                return

            x_val, y_val, z_val = float(vec[0]), float(vec[1]), float(vec[2])

            # Helper: ensure axis child exists
            def set_axis(axis, value):
                try:
                    node = getattr(param_elem, axis)
                except AttributeError:
                    # create missing axis element
                    node = XmlElement(axis, attributes={}, parent=param_elem)
                    param_elem.add_child(node)
                node.cdata = f"{value:.16g}"

            set_axis('x', x_val)
            set_axis('y', y_val)
            set_axis('z', z_val)


        # --- Access beamline and objects -------------------------------------------------
        # template_work can be XmlElement root or ElementTree-like. Try to get root if needed.
        root_elem = template_work.getroot() if hasattr(template_work, 'getroot') else template_work
        beam = find_beamline(root_elem)
        #print("Beamline:", beam)
        #print("Objects:", [o.get_attribute('name') for o in beam.get_elements('object')])

        if beam is None:
            raise ValueError("No beamline element found in template_work")
            return result

        objects_list = find_objects(beam)
        # map name -> XmlElement object
        objects = {}
        for obj in objects_list:
            # object name is stored as attribute 'name'
            name = obj.get_attribute('name')
            if name:
                objects[name] = obj

        # --- Build index -> (component, property) mapping -------------------------------
        index_to_comp_prop = {}
        for i, key in enumerate(self.param_list):
            if isinstance(key, (list, tuple)) and len(key) == 2:
                comp, prop = key
            else:
                if isinstance(key, str) and '.' in key:
                    comp, prop = key.split('.', 1)
                else:
                    comp, prop = (key, '')
            index_to_comp_prop[i] = (comp, prop)

        # --- Error detection & canonicalization helpers --------------------------------
        def is_error_property(prop_name: str) -> bool:
            if not prop_name:
                return False
            return prop_name.endswith('error')

        # --- Read baseline pos/dirs from XML for objects that have worldPosition/worldXdirection ---
        baseline = {}
        for name, obj in objects.items():
            wp = find_param(obj, 'worldPosition')
            dx = find_param(obj, 'worldXdirection')
            dy = find_param(obj, 'worldYdirection')
            dz = find_param(obj, 'worldZdirection')
            if wp is None or dx is None:
                raise ValueError(f"Missing worldPosition or worldXdirection for {name}; skipping baseline for it.")
                continue
            try:
                pos = parse_vec_xml(wp)
            except Exception:
                raise ValueError(f"Failed to parse worldPosition for {name}; skipping.")
                continue
            # parse directions with fallbacks
            dirs = []
            for d, fallback in ((dx, (1.0, 0.0, 0.0)), (dy, (0.0, 1.0, 0.0)), (dz, (0.0, 0.0, 1.0))):
                if d is not None:
                    try:
                        dirs.append(parse_vec_xml(d))
                    except Exception:
                        raise ValueError(f"Failed to parse direction param for {name}; using fallback.")
                        dirs.append(fallback)
                else:
                    dirs.append(fallback)
            baseline[name] = {'pos': tuple(pos), 'dirs': tuple(dirs)}

        # --- Write scalar (non-error) params into XML and fill param_container -----------
        for idx, (comp, prop) in index_to_comp_prop.items():
            val = parameters[idx].item()
            # store in result container under a canonical key
            keyname = f"{comp}.{prop}" if prop else comp
            result['param_container'][keyname] = val
            if prop and comp not in self.manual_transform_plane:
                obj = objects.get(comp)
                if obj is None:
                    # component not in XML; skip silently
                    continue
                node = find_param(obj, prop)
                if node is None:
                    raise ValueError(f"Parameter {prop} not found for component {comp}")
                    continue
                # write value as cdata; keep int-like floats as ints
                if isinstance(val, float) and float(val).is_integer():
                    node.cdata = str(int(val))
                else:
                    node.cdata = str(val)

        # --- For each baseline object: collect error params from tensor and apply rigid transform
        for comp, base in baseline.items():
            # Collect errors for this component
            errs = {}
            for idx, (c, prop) in index_to_comp_prop.items():
                if c != comp:
                    continue
                if not prop:
                    continue
                if not is_error_property(prop):
                    continue
                val = float(parameters[idx].item())
                key = prop
                errs[key] = val

            # skip if no meaningful errors
            if not any(abs(v) > 1e-15 for v in errs.values()):
                continue

            base_pos = np.array(base['pos'], dtype=float)         # shape (3,)
            base_dirs = np.vstack(base['dirs']).astype(float)     # shape (3,3) - rows X,Y,Z

            try:
                new_pos_arr, new_dirs_arr = apply_rigid_transform(base_pos, base_dirs, errs)
                #print("out", base_pos==new_pos_arr, base_dirs==new_dirs_arr)
            except Exception:
                print(f"apply_rigid_transform failed for component {comp} with errs={errs}")
                continue

            # Convert to plain python tuples
            new_pos = tuple(float(x) for x in new_pos_arr.tolist())
            # assume new_dirs_arr shape (3,3) rows correspond to X,Y,Z directions
            new_dirs = tuple(tuple(float(x) for x in row.tolist()) for row in new_dirs_arr)

            # write back into XML
            obj = objects.get(comp)
            if obj is None:
                continue
            wp_elem = find_param(obj, 'worldPosition')
            if wp_elem is not None:
                write_vec_xml(wp_elem, new_pos)
            for i, pid in enumerate(['worldXdirection', 'worldYdirection', 'worldZdirection']):
                elem = find_param(obj, pid)
                if elem is not None and i < len(new_dirs):
                    write_vec_xml(elem, new_dirs[i])
            #if self.verbose:
            #    print(f"{comp}: errs={errs} -> pos={new_pos}, dirs={new_dirs}")

        # --- Call the backend using the mutated XML template -----------------------------
        ray_output_all_planes = self.ray_backend.run(raypyng_rml=raypyng_rml_work,
                                                    exported_planes=self.exported_planes)
        result['ray_output'] = ray_output_all_planes

        # apply optional transform to each exported plane
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

