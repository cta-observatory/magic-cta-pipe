from compare_hillas_stereo_params import compare_hillas_stereo_parameters
from pathlib import Path

config = Path("compare_params_config.yaml").absolute()

def test_compare_hillas_stereo_params():
	list1 = compare_hillas_stereo_parameters(config_file = str(config), params_key="events/parameters", plot_image=True)
	assert list1 == [False]*len(list1)
