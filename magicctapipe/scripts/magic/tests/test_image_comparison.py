from image_comparison import image_comparison
from pathlib import Path

config=Path("image_comp_2_config.yaml").absolute()

def test_image_comparison():
	list1 = image_comparison(config_file = str(config), mode = "use_ids_config", tel_id=2)
	assert list1 == []