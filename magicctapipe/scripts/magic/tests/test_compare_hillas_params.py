from compare_hillas_params import compare_hillas_parameters
from pathlib import Path

config = Path("compare_hillas_config.yaml").absolute()

def test_compare_hillas_params():
	list1 = compare_hillas_parameters(config_file = str(config), hillas_key="events/parameters", subrun=5093711, plot_image=True)
	assert list1 == [False]*len(list1)
