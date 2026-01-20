from collections import OrderedDict
import h5py
import numpy as np
import logging


from ray_tools.simulation.data_tools import EfficientRandomRayDatasetGenerator
from ray_tools.base.transform import XYHistogram

def test_generator_fixed(tmp_path, dummy_engine, sampler):
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
    param_limit_dict:OrderedDict[str, tuple[float, float]] = OrderedDict([
        ("p1", (0.0, 1.0)),
        ("p2", (0.0, 1.0)),
    ])

    gen = EfficientRandomRayDatasetGenerator(
        engine=dummy_engine,
        sampler=sampler,
        h5_datadir=str(outdir),
        param_limit_dict=param_limit_dict,
        exported_planes=["plane1"],
        h5_basename="raw",
        h5_max_size=2,
        fixed_output_size=True,
        transform=XYHistogram(10, (0,10), (0,5)),
    )

    gen.generate(h5_idx=0)

    h5path = outdir / "raw_0.h5"
    assert h5path.exists()

    with h5py.File(h5path, "r") as f:
        # top-level groups
        assert "params" in f
        assert "meta" in f
        assert "ray_output" in f

        params_grp = f["params"]
        assert isinstance(params_grp, h5py.Group)
        assert "values" in params_grp
        assert "limits" in params_grp
        assert "names" in params_grp
        values_dataset = params_grp["values"]
        assert isinstance(values_dataset, h5py.Dataset)
        assert values_dataset.shape == (2, 2)  # 2 samples, 2 params

        assert "limits" in params_grp
        limits_dataset = params_grp["limits"]
        assert isinstance(limits_dataset, h5py.Dataset)
        assert limits_dataset.shape == (2, 2)  # 2 params, each with (min, max) 
        assert limits_dataset[0,0] == 0.0
        assert limits_dataset[0,1] == 1.0

        # plane group
        ray_output_grp = f["ray_output"]
        assert isinstance(ray_output_grp, h5py.Group)
        assert "plane1" in ray_output_grp

        # keys expected from DummyEngine simple variant
        plane_grp = f["ray_output/plane1"]
        assert isinstance(plane_grp, h5py.Group)
        assert "n_rays" in plane_grp
        assert "arr1d" in plane_grp
        assert "arr2d" in plane_grp

        # n_rays: scalar -> stored as 1 row x 1 col; offsets should index it
        n_rays_data = plane_grp['n_rays']#_read_group_data(f, "plane1", "n_rays", rec_idx=0)
        assert isinstance(n_rays_data, h5py.Dataset)
        n_rays_data = np.asarray(n_rays_data)
        logging.info(f"n_rays_data: {n_rays_data}") 
        
        # n_rays came as scalar 10, may be stored as shape (1,1)
        assert n_rays_data.size >= 1
        assert int(n_rays_data.reshape(-1)[0]) == 10

        # arr1d: original np.arange(5) -> stored as (5,1)
        arr1d = plane_grp['arr1d']
        assert isinstance(arr1d, h5py.Dataset)
        arr1d = np.asarray(arr1d)
        # flatten to compare values
        assert arr1d.shape == (2,5)
        assert np.array_equal(arr1d[0].flatten(), np.arange(5))

        # arr2d: original shape (2,3) -> stored as (2,3)
        arr2d = plane_grp["arr2d"]
        arr2d = np.asarray(arr2d)
        assert arr2d.shape == (2, 2, 3)
        assert np.array_equal(arr2d[0].reshape(-1), np.arange(6))