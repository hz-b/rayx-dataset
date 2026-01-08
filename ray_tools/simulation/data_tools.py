from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Iterable

import h5py
import numpy as np
import torch

from ..base import RayTransformType
from ..base.engine import Engine
from ..base.backend import RayOutput


_RAY_FIELDS = ("x_loc", "y_loc", "z_loc", "x_dir", "y_dir", "z_dir", "energy")


def _as_numpy_1d(x: Any, dtype=np.float32) -> np.ndarray:
    """Convert torch/np/scalar to 1D numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)

    x = np.asarray(x)
    if x.ndim != 1:
        x = x.reshape(-1)
    return x.astype(dtype, copy=False)


def _rayoutput_len(ro: RayOutput) -> int:
    # assume at least one of these exists
    for f in ("x_loc", "y_loc", "energy"):
        if hasattr(ro, f):
            return int(_as_numpy_1d(getattr(ro, f)).shape[0])
    raise ValueError("RayOutput has none of expected fields to infer length.")


class EfficientRandomRayDatasetGenerator:
    """
    Efficient HDF5 writer for your new Engine API.

    H5 layout (single file):
      /params/values         float32 [N, P]
      /params/names          utf-8 string [P]
      /meta/idx_sample       int64 [N]
      /meta/idx_sub          utf-8 string [N]

      /ray_output/<plane>/offsets   int64 [N+1]
      /ray_output/<plane>/<field>  float32 [total_rays]

    Where rays for sample i and plane p are:
      s = offsets[i]; e = offsets[i+1]
      field[s:e]
    """

    def __init__(
        self,
        engine: Engine,
        parameter_sampler: Callable[[], dict],
        h5_datadir: str,
        param_list: list[str],
        exported_planes: list[str],
        h5_basename: str = "raw",
        h5_max_size: int = 1000,   # dataset samples (idx_sample)
        device: torch.device | None = None,
        params_dtype: np.dtype = np.float32,
        rays_dtype: np.dtype = np.float32,
        chunk_rows: int = 256,     # chunking along N for params/meta
        chunk_rays: int = 1_000_000,  # chunking along total_rays
        compress: str | None = "lzf",  # "lzf" is fast; "gzip" compresses more but slower
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
        self.rays_dtype = rays_dtype
        self.chunk_rows = int(chunk_rows)
        self.chunk_rays = int(chunk_rays)
        self.compress = compress

        os.makedirs(self.h5_datadir, exist_ok=True)

    @staticmethod
    def build_parameter_sampler(
        sampler_func: Callable[[], torch.Tensor],
        idx_sub: list[str],
        transform: list[RayTransformType | None] | RayTransformType | None,
    ) -> Callable[[], dict]:
        """
        sampler_func returns torch.Tensor [n] or [m,n].
        idx_sub must have length m (m=1 common).
        transform can be singleton or list length m.
        """
        def _sampler() -> dict:
            params = sampler_func()
            if params.ndim == 1:
                params = params.unsqueeze(0)
            if params.ndim != 2:
                raise ValueError(f"sampler_func must return [n] or [m,n]. Got {tuple(params.shape)}")

            m = params.shape[0]
            if len(idx_sub) != m:
                raise ValueError(f"idx_sub length {len(idx_sub)} must match m={m}")

            if transform is None or not isinstance(transform, list):
                t_list = [transform] * m
            else:
                if len(transform) != m:
                    raise ValueError(f"transform length {len(transform)} must match m={m}")
                t_list = transform

            return {"parameters": params, "idx_sub": list(idx_sub), "transform": t_list}

        return _sampler

    def generate(self, h5_idx: int, batch_size: int = -1) -> None:
        """
        Writes one H5 file.
        batch_size is in units of dataset samples (idx_sample).
        """
        if batch_size == -1:
            batch_size = self.h5_max_size

        h5_path = os.path.join(self.h5_datadir, f"{self.h5_basename}_{h5_idx}.h5")
        with h5py.File(h5_path, "w") as f:
            # --- create top-level groups ---
            params_grp = f.require_group("params")
            meta_grp = f.require_group("meta")
            ray_grp = f.require_group("ray_output")

            # --- params datasets (resizable) ---
            P = len(self.param_list)
            params_values = params_grp.create_dataset(
                "values",
                shape=(0, P),
                maxshape=(None, P),
                dtype=self.params_dtype,
                chunks=(max(1, min(self.chunk_rows, 1024)), P),
                compression=self.compress,
            )
            params_names = params_grp.create_dataset(
                "names",
                data=np.array(self.param_list, dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )

            # --- meta datasets (resizable) ---
            idx_sample_ds = meta_grp.create_dataset(
                "idx_sample",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(max(1, min(self.chunk_rows, 4096)),),
                compression=self.compress,
            )
            idx_sub_ds = meta_grp.create_dataset(
                "idx_sub",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype("utf-8"),
                chunks=(max(1, min(self.chunk_rows, 4096)),),
                compression=self.compress,
            )

            # --- per-plane ray datasets (flat + offsets) ---
            plane_ds = {}
            for plane in self.exported_planes:
                pgrp = ray_grp.require_group(plane)

                offsets = pgrp.create_dataset(
                    "offsets",
                    shape=(1,),          # offsets[0] = 0
                    maxshape=(None,),
                    dtype=np.int64,
                    chunks=(max(1, min(self.chunk_rows, 4096)),),
                    compression=self.compress,
                )
                offsets[0] = 0

                fields = {}
                for field in _RAY_FIELDS:
                    fields[field] = pgrp.create_dataset(
                        field,
                        shape=(0,),
                        maxshape=(None,),
                        dtype=self.rays_dtype,
                        chunks=(max(1024, min(self.chunk_rays, 10_000_000)),),
                        compression=self.compress,
                    )

                plane_ds[plane] = {"offsets": offsets, "fields": fields, "total_rays": 0}

            # --- main loop ---
            N = 0  # number of records (engine runs) written so far (NOT idx_sample)
            idx_total_sample = 0

            while idx_total_sample < self.h5_max_size:
                end = min(idx_total_sample + batch_size, self.h5_max_size)

                # Collect a batch in "records" (each idx_sub entry is a record)
                rows: list[torch.Tensor] = []
                rec_idx_sample: list[int] = []
                rec_idx_sub: list[str] = []
                rec_transforms: list[RayTransformType | None] = []

                for idx_sample in range(idx_total_sample, end):
                    sampled = self.parameter_sampler()
                    if not all(k in sampled for k in ("parameters", "idx_sub", "transform")):
                        raise KeyError("parameter_sampler must return keys: 'parameters', 'idx_sub', 'transform'")

                    params: torch.Tensor = sampled["parameters"]
                    if params.ndim == 1:
                        params = params.unsqueeze(0)
                    if params.ndim != 2:
                        raise ValueError(f"'parameters' must be [m,n] (or [n]). Got {tuple(params.shape)}")

                    idx_sub = list(sampled["idx_sub"])
                    transform = sampled["transform"]
                    m = params.shape[0]

                    if len(idx_sub) != m:
                        raise ValueError(f"len(idx_sub)={len(idx_sub)} must equal m={m}")

                    if transform is None or not isinstance(transform, list):
                        transforms_list = [transform] * m
                    else:
                        if len(transform) != m:
                            raise ValueError(f"len(transform)={len(transform)} must equal m={m}")
                        transforms_list = transform

                    if self.device is not None:
                        params = params.to(self.device)

                    rows.append(params)
                    rec_idx_sample.extend([idx_sample] * m)
                    rec_idx_sub.extend(idx_sub)
                    rec_transforms.extend(transforms_list)

                if not rows:
                    break

                batch_parameters = torch.cat(rows, dim=0)  # [B, P]
                results = self.engine.run(parameters=batch_parameters, transforms=rec_transforms)
                if len(results) != batch_parameters.shape[0]:
                    raise RuntimeError(f"Engine returned {len(results)} results, expected {batch_parameters.shape[0]}")

                B = len(results)

                # --- append params + meta (record-wise) ---
                params_values.resize((N + B, P))
                idx_sample_ds.resize((N + B,))
                idx_sub_ds.resize((N + B,))

                params_values[N:N + B, :] = _as_numpy_1d(batch_parameters.detach().cpu(), dtype=self.params_dtype).reshape(B, P)
                idx_sample_ds[N:N + B] = np.asarray(rec_idx_sample, dtype=np.int64)
                idx_sub_ds[N:N + B] = np.asarray(rec_idx_sub, dtype=object)

                # --- append rays per record and plane ---
                for i, res in enumerate(results):
                    if not isinstance(res, dict) or "ray_output" not in res:
                        raise ValueError("Engine result must be dict with key 'ray_output'")

                    ray_out = res["ray_output"]
                    for plane in self.exported_planes:
                        ro = ray_out[plane]
                        if not isinstance(ro, RayOutput):
                            raise TypeError(f"ray_output[{plane}] must be RayOutput, got {type(ro)}")

                        L = _rayoutput_len(ro)

                        pd = plane_ds[plane]
                        start = pd["total_rays"]
                        end_r = start + L

                        # resize & write each field slice
                        for field in _RAY_FIELDS:
                            arr = _as_numpy_1d(getattr(ro, field), dtype=self.rays_dtype)
                            if arr.shape[0] != L:
                                raise ValueError(f"{plane}.{field} length mismatch: {arr.shape[0]} vs {L}")

                            ds = pd["fields"][field]
                            ds.resize((end_r,))
                            ds[start:end_r] = arr

                        # update offsets: append new total_rays after this record
                        pd["total_rays"] = end_r
                        off = pd["offsets"]
                        off.resize((N + i + 2,))  # need offsets up to record index (N+i)+1
                        off[N + i + 1] = end_r

                N += B
                idx_total_sample = end

                f.flush()
                print(f"Wrote {N} records so far to {h5_path}")

            print(f"Done. Records: {N}, params shape: {params_values.shape}")


# Example slicing helper (optional, for reading later)
def get_sample_rays(h5_file: h5py.File, plane: str, i: int) -> dict[str, np.ndarray]:
    offsets = h5_file[f"ray_output/{plane}/offsets"]
    s = int(offsets[i])
    e = int(offsets[i + 1])
    out = {}
    for field in _RAY_FIELDS:
        out[field] = h5_file[f"ray_output/{plane}/{field}"][s:e]
    return out

