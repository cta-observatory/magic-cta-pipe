import os
from image_comparison import image_comparison
from pathlib import Path
import yaml

import pytest

test_data = Path(os.getenv("MAGIC_TEST_DATA", "test_data")).absolute()
test_calibrated_real_dir = test_data / "real/calibrated"
test_images_real_dir = test_data / "real/images"

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


@pytest.mark.parametrize(
    "dataset_calibrated, dataset_images",
    [
        (calibrated_file, image_file)
        for calibrated_file in test_calibrated_real
        for image_file in test_images_mars_real
    ],
)
def test_image_comparison(dataset_calibrated, dataset_images, tmp_path):

    config_image = {
        "input_files": {
            "magic_cta_pipe": {"M1": dataset_calibrated, "M2": dataset_calibrated},
            "mars": dataset_images,
        },
        "output_files": {"file_path": tmp_path},
        "event_list": [1961, 1962, 1964, 1965, 2001],
        "save_only_when_differences": True,
    }

    config_image_file = str(tmp_path / "image_comparison_config.yaml")

    with open(config_image_file, "w") as outfile:
        yaml.dump(config_image, outfile, default_flow_style=False)

    if "_M1_" in dataset_calibrated:
        list_image = image_comparison(
            config_file=config_image_file, mode="use_ids_config", tel_id=1
        )
    else:
        list_image = image_comparison(
            config_file=config_image_file, mode="use_ids_config", tel_id=2
        )

    assert list_image == []
