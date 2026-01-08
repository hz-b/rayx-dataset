import unittest

import omegaconf
from hydra.utils import instantiate
from hydra import initialize, compose

from ray_nn.nn.xy_hist_data_models import HistSurrogateEngine
from ray_optim.plot import Plot
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.engine import RayEngine
from ray_tools.base.parameter import MutableParameter, NumericalParameter, RandomParameter, RayParameterContainer
from ray_tools.base.transform import Histogram, RayTransformConcat, MultiLayer
from ray_tools.base.utils import RandomGenerator
from datasets.metrix_simulation.config_ray_emergency_surrogate import PARAM_CONTAINER_FUNC as params


class RayBackendTest(unittest.TestCase):

    def setUp(self):
        self.exported_planes = ["ImagePlane"]
        self.n_rays = 100

        self.rg = RandomGenerator(seed=42)
        self.dist_layers = [-25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25., 30.]
        #self.transform = RayTransformConcat({
        #    'ml': MultiLayer(dist_layers=self.dist_layers,
        #                     copy_directions=False,
        #                     transform=Histogram(n_bins=256)),
        #    'hist': Histogram(n_bins=1024),

        #})
        self.transform = MultiLayer([0.], copy_directions=False)

        self.engine = RayEngine(rml_basefile='rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn_1.15.rml',
                                exported_planes=self.exported_planes,
                                ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                                  docker_container_name='ray-ui-service-test',
                                                                  dockerfile_path='ray_docker/rayui',
                                                                  ray_workdir='/dev/shm/ray-workdir',
                                                                  verbose=True),
                                num_workers=-1,
                                as_generator=False)
        self.surrogate_engine = HistSurrogateEngine()

    def param_func(self):
        return RayParameterContainer([
            (self.engine.template.U41_318eV.numberRays, NumericalParameter(value=self.n_rays)),
            (
                self.engine.template.E2.translationYerror,
                RandomParameter(value_lims=(-6., 6.), rg=self.rg))
        ])
    def test_param_func2(self):
        param_container = params()
        param_container = RayParameterContainer({label: NumericalParameter((entry.value_lims[1]-entry.value_lims[0]) / 2 + entry.value_lims[0]) if isinstance(entry, MutableParameter) else entry for label, entry in param_container.items()})
        normalized_param_container = Plot.normalize_parameters(param_container, params())
        print("test")
        
    def test_surrogate(self):
        n_examples = 1
        param_containers = [self.param_func() for _ in range(n_examples)]
        #transforms = {exported_plane: self.transform for exported_plane in self.exported_planes}

        result = self.engine.run(param_containers=param_containers,
                                 transforms=self.transform)
        result2 = self.surrogate_engine.run(param_containers, self.transform)
        self.assertTrue(type(result) is list)


    def test_output_type_single_output(self):
        # For single output the output should be a list
        n_examples = 1

        result = self.engine.run(param_containers=[self.param_func() for _ in range(n_examples)],
                                 transforms={exported_plane: self.transform for exported_plane in self.exported_planes})
        self.assertTrue(type(result) is list)

    def test_output_type_multiple_outputs(self):
        # For more than one output, the outputs should be a list of dictionaries
        n_examples = 2
        result = self.engine.run(param_containers=[self.param_func() for _ in range(n_examples)],
                                 transforms={exported_plane: self.transform for exported_plane in self.exported_planes})
        self.assertTrue(type(result) is list)

    def test_param_dic(self):
        # check if all parameters are contained in the output
        n_examples = 2
        result = self.engine.run(param_containers=[self.param_func() for _ in range(n_examples)],
                                 transforms={exported_plane: self.transform for exported_plane in self.exported_planes})
        param_dic = result[1]['param_container_dict']
        for key in self.param_func().to_value_dict().keys():
            self.assertTrue(key in param_dic.keys())

    def test_dist_layers(self):
        # Check if all distance layers are present
        # Also check if all n_rays, x_lim and y_lim is present in each layer
        n_examples = 2
        result = self.engine.run(param_containers=[self.param_func() for _ in range(n_examples)],
                                 transforms={exported_plane: self.transform for exported_plane in self.exported_planes})
        for dist in self.dist_layers:
            self.assertTrue(str(dist) in result[0]['ray_output']['ImagePlane']['ml'].keys())
            self.assertTrue('n_rays' in result[0]['ray_output']['ImagePlane']['ml'][str(dist)].keys())
            self.assertTrue('x_lims' in result[0]['ray_output']['ImagePlane']['ml'][str(dist)].keys())
            self.assertTrue('y_lims' in result[0]['ray_output']['ImagePlane']['ml'][str(dist)].keys())
            self.assertTrue('histogram' in result[0]['ray_output']['ImagePlane']['ml'][str(dist)].keys())
    
    def test_with_initialize(self) -> None:
        with initialize(version_base=None, config_path="../sub_projects/ray_optimization/conf"):
            # config is relative to a module
            cfg = compose(config_name="config")
            print(omegaconf.OmegaConf.to_yaml(cfg))
            ray_optimization = instantiate(cfg)
            ray_optimization.setup_target()

if __name__ == '__main__':
    unittest.main()
