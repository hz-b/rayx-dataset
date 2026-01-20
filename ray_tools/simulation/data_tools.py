from __future__ import annotations

from collections import OrderedDict
import os
from collections.abc import Callable
from typing import Any

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from ..base.transform import RayTransform
from ..base.engine import Engine


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch / scalar / array-like to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


class EfficientRandomRayDatasetGenerator:
    """
    Generic HDF5 writer for Engine outputs.

    HDF5 layout (variable-length mode):

      /params/values        [N, P]
      /params/names         [P]
      /meta/idx_sample      [N]
      /meta/idx_sub         [N]

      /ray_output/<plane>/<key>/
          data     [total_rows, cols]
          offsets  [N+1]

    HDF5 layout (fixed_output_size=True):

      /params/values        [N, P]
      /params/names         [P]
      /meta/idx_sample      [N]
      /meta/idx_sub         [N]

      /ray_output/<plane>/<key>    <- dataset [N, rows_per_record, cols]

    Offsets always index rows per record in variable-length mode.
    """

    def __init__(
        self,
        engine: Engine,
        sampler: Callable[[int], torch.Tensor],
        transform: RayTransform | None,
        h5_datadir: str,
        param_limit_dict: OrderedDict[str, tuple[float, float]],
        exported_planes: list[str],
        h5_basename: str = "raw",
        h5_max_size: int = 1000,
        device: torch.device | None = None,
        params_dtype: np.dtype = np.dtype(np.float32),
        compress: str | None = "lzf",
        fixed_output_size: bool = False,
    ) -> None:
        self.engine = engine
        self.sampler = sampler
        self.transform = transform
        self.h5_datadir = h5_datadir
        self.param_limit_dict = param_limit_dict
        self.exported_planes = list(exported_planes)
        self.h5_basename = h5_basename
        self.h5_size = h5_max_size
        self.device = device
        self.params_dtype = params_dtype
        self.compress = compress
        self.fixed_output_size = bool(fixed_output_size)

        os.makedirs(self.h5_datadir, exist_ok=True)

    def generate(self, h5_idx: int, batch_size: int = -1) -> None:
        if batch_size == -1:
            batch_size = self.h5_size

        path = os.path.join(self.h5_datadir, f"{self.h5_basename}_{h5_idx}.h5")

        with h5py.File(path, "w") as f:
            # ---------------- params ----------------
            params_grp = f.create_group("params")
            meta_grp = f.create_group("meta")
            ray_grp = f.create_group("ray_output")

            P = len(self.param_limit_dict)

            params_values = params_grp.create_dataset(
                "values",
                shape=(self.h5_size, P),
                dtype=self.params_dtype,
                compression=self.compress,
            )

            names = np.array(list(self.param_limit_dict.keys()), dtype=object)
            limits = np.array(list(self.param_limit_dict.values()), dtype=np.float64)  # shape (P, 2)


            params_grp.create_dataset(
                "limits",
                data=limits,
                dtype=np.float64,
            )

            params_grp.create_dataset(
                "names",
                data=names,
                dtype=h5py.string_dtype("utf-8"),
            )

            # create plane groups
            plane_groups = {
                plane: ray_grp.create_group(plane)
                for plane in self.exported_planes
            }

            idx_total_sample = 0

            with tqdm(
                total=self.h5_size,
                desc=f"Writing {os.path.basename(path)}",
                unit="sample",
                leave=False
            ) as pbar:

                while idx_total_sample < self.h5_size:
                    end = min(idx_total_sample + batch_size, self.h5_size)

                    batch_params = self.sampler(end - idx_total_sample)
                    results = self.engine.run(batch_params, self.transform)

                    for i, res in enumerate(results):
                        params_values[i + idx_total_sample] = _to_numpy(batch_params[i])
                        ro_all = res["ray_output"]

                        for plane, ro in ro_all.items():
                            if plane not in plane_groups:
                                continue

                            pgrp = plane_groups[plane]

                            if not isinstance(ro, dict):
                                raise TypeError(
                                    f"ray_output[{plane}] must be dict, got {type(ro)}"
                                )

                            for key, value in ro.items():
                                if not hasattr(value, "shape"):
                                    value = np.array(value)

                                value_np = _to_numpy(value)

                                if key in pgrp:
                                    dset = pgrp[key]
                                    assert isinstance(dset, h5py.Dataset)
                                    dset[i + idx_total_sample] = value_np
                                else:
                                    dset = pgrp.create_dataset(
                                        key,
                                        shape=(self.h5_size, *value_np.shape),
                                        maxshape=(None, *value_np.shape),
                                        dtype=value_np.dtype,
                                        compression=self.compress,
                                    )
                                    dset[0] = value_np

                    written = end - idx_total_sample
                    idx_total_sample = end
                    pbar.update(written)
