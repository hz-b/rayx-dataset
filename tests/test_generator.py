import h5py
import numpy as np
import torch

import pytest

from ray_tools.simulation.data_tools import EfficientRandomRayDatasetGenerator


def _read_group_data(f, plane, key, rec_idx=0):
    """Helper: return ndarray for a given plane/key and record index."""
    grp = f[f"ray_output/{plane}/{key}"]
    offsets = grp["offsets"]
    s = int(offsets[rec_idx])
    e = int(offsets[rec_idx + 1])
    return grp["data"][s:e]


def test_generator_basic(tmp_path, dummy_engine, sampler):
    """
    Basic integration test:
      - DummyEngine (variant 'simple') returns:
          n_rays: scalar 10
          arr1d: 1D np.arange(5)
          arr2d: 2x3 np.arange(6).reshape(2,3)
      - Expect groups written under ray_output/plane1/<key>/data and offsets
    """
    outdir = tmp_path / "out"
    outdir.mkdir()

    gen = EfficientRandomRayDatasetGenerator(
        engine=dummy_engine,
        parameter_sampler=sampler,
        h5_datadir=str(outdir),
        param_list=["p1", "p2"],
        exported_planes=["plane1"],
        h5_basename="raw",
        h5_max_size=1,
    )

    gen.generate(h5_idx=0)

    h5path = outdir / "raw_0.h5"
    assert h5path.exists()

    with h5py.File(h5path, "r") as f:
        # top-level groups
        assert "params" in f
        assert "meta" in f
        assert "ray_output" in f

        # plane group
        assert "plane1" in f["ray_output"]

        # keys expected from DummyEngine simple variant
        plane_grp = f["ray_output/plane1"]
        assert "n_rays" in plane_grp
        assert "arr1d" in plane_grp
        assert "arr2d" in plane_grp

        # n_rays: scalar -> stored as 1 row x 1 col; offsets should index it
        n_rays_data = _read_group_data(f, "plane1", "n_rays", rec_idx=0)
        # n_rays came as scalar 10, may be stored as shape (1,1)
        assert n_rays_data.size >= 1
        assert int(n_rays_data.reshape(-1)[0]) == 10

        # arr1d: original np.arange(5) -> stored as (5,1)
        arr1d = _read_group_data(f, "plane1", "arr1d", rec_idx=0)
        # flatten to compare values
        arr1d_flat = np.asarray(arr1d).reshape(-1)
        assert arr1d_flat.shape[0] == 5
        assert np.array_equal(arr1d_flat, np.arange(5))

        # arr2d: original shape (2,3) -> stored as (2,3)
        arr2d = _read_group_data(f, "plane1", "arr2d", rec_idx=0)
        arr2d = np.asarray(arr2d)
        assert arr2d.shape == (2, 3)
        assert np.array_equal(arr2d.reshape(-1), np.arange(6))


def test_generator_hist_variant(tmp_path, dummy_engine_variant, sampler):
    """
    Variant test where DummyEngine returns:
      histogram: torch.ones((2,4))
      meta: scalar 42

    Verify hist data (2x4) and meta scalar are written.
    """
    outdir = tmp_path / "out_hist"
    outdir.mkdir()
    print(outdir)

    gen = EfficientRandomRayDatasetGenerator(
        engine=dummy_engine_variant,
        parameter_sampler=sampler,
        h5_datadir=str(outdir),
        param_list=["p1", "p2"],
        exported_planes=["plane1"],
        h5_basename="raw",
        h5_max_size=1,
    )

    gen.generate(h5_idx=0)

    h5path = outdir / "raw_0.h5"
    assert h5path.exists()

    with h5py.File(h5path, "r") as f:
        plane_grp = f["ray_output/plane1"]
        assert "histogram" in plane_grp
        assert "meta" in plane_grp

        # histogram reading
        hist = _read_group_data(f, "plane1", "histogram", rec_idx=0)
        # hist was torch.ones((2,4)) -> stored as (2,4)
        hist = np.asarray(hist)
        assert hist.shape == (2, 4)
        assert np.all(hist == 1)

        # meta scalar
        meta = _read_group_data(f, "plane1", "meta", rec_idx=0)
        assert meta.size is not None
        assert meta.size >= 1
        # value was integer 42
        assert int(np.asarray(meta).reshape(-1)[0]) == 42
