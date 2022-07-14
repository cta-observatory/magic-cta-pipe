import os
from image_comparison import image_comparison
from pathlib import Path
import yaml

import pytest

test_data = Path(os.getenv("MAGIC_TEST_DATA", "test_data")).absolute()
test_calibrated_real_dir = test_data / "real/calibrated"
test_images_real_dir = test_data / "real/images"
test_calibrated_simulated_dir = test_data / "simulated/calibrated"
test_images_simulated_dir = test_data / "simulated/images"

test_calibrated_real = [
    test_calibrated_real_dir / "20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root",
    test_calibrated_real_dir / "20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root",
    test_calibrated_real_dir / "20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root",
    test_calibrated_real_dir / "20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root",
]

test_images_mars_real = [
    test_images_real_dir / "20210314_M1_05095172.001_I_CrabNebula-W0.40+035.h5",
    test_images_real_dir / "20210314_M1_05095172.002_I_CrabNebula-W0.40+035.h5",
    test_images_real_dir / "20210314_M2_05095172.001_I_CrabNebula-W0.40+035.h5",
    test_images_real_dir / "20210314_M2_05095172.002_I_CrabNebula-W0.40+035.h5",
]

test_calibrated_simulated = [
    test_calibrated_simulated_dir / "GA_M1_za35to50_8_824318_Y_w0.root",
    test_calibrated_simulated_dir / "GA_M1_za35to50_8_824319_Y_w0.root",
    test_calibrated_simulated_dir / "GA_M2_za35to50_8_824318_Y_w0.root",
    test_calibrated_simulated_dir / "GA_M2_za35to50_8_824319_Y_w0.root",
]

test_images_mars_simulated = [
    test_images_simulated_dir / "GA_M1_za35to50_8_824318_I_w0.h5",
    test_images_simulated_dir / "GA_M1_za35to50_8_824319_I_w0.h5",
    test_images_simulated_dir / "GA_M2_za35to50_8_824318_I_w0.h5",
    test_images_simulated_dir / "GA_M2_za35to50_8_824319_I_w0.h5",
]

file_list = []

for i in range(len(test_calibrated_simulated)):
    file_list.append((test_calibrated_simulated[i], test_images_mars_simulated[i]))

for i in range(len(test_calibrated_real)):
    file_list.append((test_calibrated_real[i], test_images_mars_real[i]))


@pytest.mark.parametrize(
    "dataset_calibrated, dataset_images",
    file_list,
)
def test_image_comparison(dataset_calibrated, dataset_images, tmp_path):

    config_image = {
        "input_files": {
            "magic_cta_pipe": {
                "M1": str(dataset_calibrated),
                "M2": str(dataset_calibrated),
            },
            "mars": str(dataset_images),
        },
        "output_files": {"file_path": str(test_data / "real/test_images")},
        "event_list": [1961, 1962, 1964, 1965, 2001],
        "save_only_when_differences": True,
    }

    config_image_file = str(tmp_path / "image_comparison_config.yaml")

    with open(config_image_file, "w") as outfile:
        yaml.dump(config_image, outfile, default_flow_style=False)

    if "_M1_" in str(dataset_calibrated):
        list_image = image_comparison(
            config_file=config_image_file, mode="use_all", tel_id=1, max_events=20
        )
    else:
        list_image = image_comparison(
            config_file=config_image_file, mode="use_all", tel_id=2, max_events=20
        )

    assert list_image == []
