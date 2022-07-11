from compare_hillas_stereo_params import compare_hillas_stereo_parameters
from pathlib import Path
from pkg_resources import resource_filename
import yaml

test_data = Path(os.getenv('MAGIC_TEST_DATA', 'test_data')).absolute()
test_calibrated_real_dir = test_data / 'real/calibrated'
test_calibrated_mars_M1_real = [
    test_calibrated_real_dir / '20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root',
]

test_calibrated_mars_M2_real = [
    test_calibrated_real_dir / '20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root',
]

# TODO: produce these files
test_superstar_mars_real = [
    test_calibrated_real_dir / '20210314_05095172_S_CrabNebula-W0.40+035.root',
]

test_calibrated_all = test_calibrated_real+test_calibrated_simulated

config = resource_filename(
	'magicctapipe', 'scripts/mars/tests/compare_params_config.yaml'
)

config_mcp_file = resource_filename(
	'magicctapipe', 'scripts/lst1_magic/config.yaml'
)

# TODO: check if this will consider all combinations or just one to one
@pytest.mark.parametrize('dataset_calibrated_M1', test_calibrated_mars_M1_real)
@pytest.mark.parametrize('dataset_calibrated_M2', test_calibrated_mars_M2_real)
@pytest.mark.parametrize('dataset_superstar', test_superstar_mars_real)
def test_compare_hillas_stereo_params(dataset_calibrated_M1, dataset_calibrated_M2, dataset_superstar):
	from magicctapipe.scripts.lst1_magic import magic_calib_to_dl1, merge_hdf_files, stereo_reconstruction

	with yaml.safe_load(open(config, "rw")) as config_file:
		config_file["MARS-input"]["MARS-path"] = dataset_superstar

	with open(config_mcp_file, 'rb') as f:
	    config_mcp = yaml.safe_load(f)

	magic_calib_to_dl1(dataset_calibrated_M1, test_data / 'real/dl1', config_mcp, True)
	magic_calib_to_dl1(dataset_calibrated_M2, test_data / 'real/dl1', config_mcp, True)

	merge_hdf_files(test_data / 'real/dl1', output_dir=test_data / 'real/dl1_merged', run_wise=True, subrun_wise=False)

	stereo_reconstruction(test_data / 'real/dl1_merged' / 'dl1_MAGIC.Run05095172.h5', test_data / 'real/dl1_stereo' , config_mcp, magic_only=True)

	with yaml.safe_load(open(config, "rw")) as config_file:
		config_file["magic-cta-pipe-input"]["MCP-path"] = test_data / 'real/dl1_stereo' / 'dl1_stereo_MAGIC.Run05095172.h5'
		config_file["MARS-input"]["MARS-path"] = dataset_superstar

	list_compare_parameters = compare_hillas_stereo_parameters(config_file = config, params_key="events/parameters", plot_image=True)
	assert list_compare_parameters == [False]*len(list_compare_parameters)
