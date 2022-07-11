from image_comparison import image_comparison
from pathlib import Path
from pkg_resources import resource_filename

test_data = Path(os.getenv('MAGIC_TEST_DATA', 'test_data')).absolute()
test_calibrated_real_dir = test_data / 'real/calibrated'
test_calibrated_real = [
    test_calibrated_real_dir / '20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root',
    test_calibrated_real_dir / '20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root',
    test_calibrated_real_dir / '20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root',
    test_calibrated_real_dir / '20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root',
]

# TODO: produce these files
test_images_mars_real = [
    test_calibrated_real_dir / '20210314_M1_05095172.001_I_CrabNebula-W0.40+035.h5',
    test_calibrated_real_dir / '20210314_M1_05095172.002_I_CrabNebula-W0.40+035.h5',
    test_calibrated_real_dir / '20210314_M2_05095172.001_I_CrabNebula-W0.40+035.h5',
    test_calibrated_real_dir / '20210314_M2_05095172.002_I_CrabNebula-W0.40+035.h5',
]

test_calibrated_all = test_calibrated_real+test_calibrated_simulated

config = resource_filename(
	'magicctapipe', 'scripts/mars/tests/image_comparison_config.yaml'
)

# TODO: check if this will consider all combinations or just one to one
@pytest.mark.parametrize('dataset_calibrated', test_calibrated_real)
@pytest.mark.parametrize('dataset_images', test_images_mars_real)
def test_image_comparison(dataset_calibrated, dataset_images):
	with yaml.safe_load(open(config, "rw")) as config_file:
		if "_M1_" in dataset_calibrated:
			config_file["input_files"]["magic_cta_pipe"]["M1"] = dataset_calibrated
			config_file["input_files"]["magic_cta_pipe"]["mars"] = dataset_images
		else if "_M2_" in dataset_calibrated:
			config_file["input_files"]["magic_cta_pipe"]["M2"] = dataset_calibrated
			config_file["input_files"]["magic_cta_pipe"]["mars"] = dataset_images
		else:
			print("File type not recognized. Exiting.")
			return False

	if "_M1_" in dataset_calibrated:
		list_image = image_comparison(config_file=config, mode = "use_ids_config", tel_id=1)
	else:
		list_image = image_comparison(config_file=config, mode = "use_ids_config", tel_id=2)
	assert list_image == []
