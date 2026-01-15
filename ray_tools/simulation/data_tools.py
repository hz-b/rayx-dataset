from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import h5py
import numpy as np
import torch

from ..base import RayTransformType
from ..base.engine import Engine


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch / scalar / array-like to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class EfficientRandomRayDatasetGenerator:
    """
    Generic HDF5 writer for Engine outputs.

    Writes *whatever* appears in `ray_output[plane]` without
    interpretation or hardcoding.

    HDF5 layout:

      /params/values        [N, P]
      /params/names         [P]
      /meta/idx_sample      [N]
      /meta/idx_sub         [N]

      /ray_output/<plane>/<key>/
          data     [total_rows, cols]
          offsets  [N+1]

    Offsets always index rows per record.
    """

    def __init__(
        self,
        engine: Engine,
        parameter_sampler: Callable[[], dict],
        h5_datadir: str,
        param_list: list[str],
        exported_planes: list[str],
        h5_basename: str = "raw",
        h5_max_size: int = 1000,
        device: torch.device | None = None,
        params_dtype: np.dtype = np.float32,
        compress: str | None = "lzf",
    ) -> None:
        self.engine = engine
        self.parameter_sampler = parameter_sampler
        self.h5_datadir = h5_datadir
        self.param_list = list(param_list)
        self.exported_planes = list(exported_planes)
        self.h5_basename = h5_basename
        self.h5_max_size = int(h5_max_size)
        self.device = device
        self.params_dtype = params_dtype
        self.compress = compress

        os.makedirs(self.h5_datadir, exist_ok=True)

    # ------------------------------------------------------------------
    # Parameter sampler helper (UNCHANGED)
    # ------------------------------------------------------------------
    @staticmethod
    def build_parameter_sampler(
        sampler_func: Callable[[], torch.Tensor],
        idx_sub: list[str],
        transform: list[RayTransformType | None] | RayTransformType | None,
    ) -> Callable[[], dict]:
        def _sampler() -> dict:
            params = sampler_func()
            if params.ndim == 1:
                params = params.unsqueeze(0)
            if params.ndim != 2:
                raise ValueError(f"Expected [n] or [m,n], got {params.shape}")

            m = params.shape[0]
            if len(idx_sub) != m:
                raise ValueError("idx_sub length mismatch")

            if transform is None or not isinstance(transform, list):
                t_list = [transform] * m
            else:
                if len(transform) != m:
                    raise ValueError("transform length mismatch")
                t_list = transform

            return {
                "parameters": params,
                "idx_sub": list(idx_sub),
                "transform": t_list,
            }

        return _sampler

    # ------------------------------------------------------------------
    # Main writer
    # ------------------------------------------------------------------
    def generate(self, h5_idx: int, batch_size: int = -1) -> None:
        if batch_size == -1:
            batch_size = self.h5_max_size

        path = os.path.join(self.h5_datadir, f"{self.h5_basename}_{h5_idx}.h5")

        with h5py.File(path, "w") as f:
            # ---------------- params ----------------
            params_grp = f.create_group("params")
            meta_grp = f.create_group("meta")
            ray_grp = f.create_group("ray_output")

            P = len(self.param_list)

            params_values = params_grp.create_dataset(
                "values",
                shape=(0, P),
                maxshape=(None, P),
                dtype=self.params_dtype,
                compression=self.compress,
            )
            params_grp.create_dataset(
                "names",
                data=np.array(self.param_list, dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )

            idx_sample_ds = meta_grp.create_dataset(
                "idx_sample",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int64,
                compression=self.compress,
            )
            idx_sub_ds = meta_grp.create_dataset(
                "idx_sub",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype("utf-8"),
                compression=self.compress,
            )

            # create plane groups
            plane_groups = {
                plane: ray_grp.create_group(plane)
                for plane in self.exported_planes
            }

            N = 0
            idx_total_sample = 0

            while idx_total_sample < self.h5_max_size:
                end = min(idx_total_sample + batch_size, self.h5_max_size)

                rows = []
                rec_idx_sample = []
                rec_idx_sub = []
                rec_transforms = []

                for idx_sample in range(idx_total_sample, end):
                    s = self.parameter_sampler()
                    params = s["parameters"]
                    if params.ndim == 1:
                        params = params.unsqueeze(0)

                    if self.device:
                        params = params.to(self.device)

                    rows.append(params)
                    rec_idx_sample.extend([idx_sample] * params.shape[0])
                    rec_idx_sub.extend(s["idx_sub"])
                    rec_transforms.extend(s["transform"])

                if not rows:
                    break

                batch_params = torch.cat(rows, dim=0)
                results = self.engine.run(batch_params, rec_transforms)

                B = len(results)

                # write params + meta
                params_values.resize((N + B, P))
                params_values[N:N + B] = batch_params.cpu().numpy()

                idx_sample_ds.resize((N + B,))
                idx_sample_ds[N:N + B] = rec_idx_sample

                idx_sub_ds.resize((N + B,))
                idx_sub_ds[N:N + B] = np.asarray(rec_idx_sub, dtype=object)

                # ---------------- ray output ----------------
                for i, res in enumerate(results):
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
                            arr = _to_numpy(value)

                            # normalize shape → (rows, cols)
                            if arr.ndim == 0:
                                arr = arr.reshape(1, 1)
                            elif arr.ndim == 1:
                                arr = arr.reshape(-1, 1)
                            else:
                                arr = arr.reshape(arr.shape[0], -1)

                            rows_k, cols = arr.shape

                            kgrp = pgrp.require_group(key)

                            if "data" not in kgrp:
                                dset = kgrp.create_dataset(
                                    "data",
                                    shape=(0, cols),
                                    maxshape=(None, cols),
                                    dtype=arr.dtype,
                                    compression=self.compress,
                                )
                                offsets = kgrp.create_dataset(
                                    "offsets",
                                    shape=(1,),
                                    maxshape=(None,),
                                    dtype=np.int64,
                                    compression=self.compress,
                                )
                                offsets[0] = 0
                            else:
                                dset = kgrp["data"]
                                offsets = kgrp["offsets"]

                            start = dset.shape[0]
                            end_k = start + rows_k

                            dset.resize((end_k, cols))
                            dset[start:end_k] = arr

                            offsets.resize((N + i + 2,))
                            offsets[N + i + 1] = end_k

                N += B
                idx_total_sample = end
                f.flush()

                print(f"Wrote {N} records → {path}")

            print(f"Done. Total records: {N}")


# ----------------------------------------------------------------------
# Generic reader helper
# ----------------------------------------------------------------------
def get_sample(h5: h5py.File, plane: str, key: str, i: int) -> np.ndarray:
    grp = h5[f"ray_output/{plane}/{key}"]
    s = grp["offsets"][i]
    e = grp["offsets"][i + 1]
    return grp["data"][s:e]

