import os
from compare_hillas_stereo_params import compare_hillas_stereo_parameters
from pathlib import Path
import yaml
import numpy as np

import pytest

test_data = Path(os.getenv("MAGIC_TEST_DATA", "test_data")).absolute()
test_calibrated_real_dir = test_data / "real/calibrated"
test_superstar_real_dir = test_data / "real/superstar"
test_calibrated_simulated_dir = test_data / "simulated/calibrated"
test_superstar_simulated_dir = test_data / "simulated/superstar"

test_calibrated_mars_M1_real = [
    test_calibrated_real_dir / "20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root",
]

test_calibrated_mars_M2_real = [
    test_calibrated_real_dir / "20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root",
]

test_superstar_mars_real = [
    test_superstar_real_dir / "20210314_05095172_S_CrabNebula-W0.40+035.root",
]

test_calibrated_mars_M1_simulated = [
    test_calibrated_simulated_dir / "GA_M1_za35to50_8_824318_Y_w0.root",
    test_calibrated_simulated_dir / "GA_M1_za35to50_8_824319_Y_w0.root",
]

test_calibrated_mars_M2_simulated = [
    test_calibrated_simulated_dir / "GA_M2_za35to50_8_824318_Y_w0.root",
    test_calibrated_simulated_dir / "GA_M2_za35to50_8_824319_Y_w0.root",
]

test_superstar_mars_simulated = [
    test_superstar_simulated_dir / "GA_za35to50_8_824318_S_w0.root",
    test_superstar_simulated_dir / "GA_za35to50_8_824319_S_w0.root",
]

file_list = []

for i in range(len(test_calibrated_mars_M1_simulated)):
    file_list.append(
        (
            test_calibrated_mars_M1_simulated[i],
            test_calibrated_mars_M2_simulated[i],
            test_superstar_mars_simulated[i],
        )
    )

for i in range(len(test_calibrated_mars_M1_real)):
    file_list.append(
        (
            test_calibrated_mars_M1_real[i],
            test_calibrated_mars_M2_real[i],
            test_superstar_mars_real[i],
        )
    )


@pytest.mark.parametrize(
    "dataset_calibrated_M1, dataset_calibrated_M2, dataset_superstar",
    file_list,
)
def test_compare_hillas_stereo_params(
    dataset_calibrated_M1, dataset_calibrated_M2, dataset_superstar, tmp_path
):
    from magicctapipe.scripts.lst1_magic import (
        magic_calib_to_dl1,
        merge_hdf_files,
        stereo_reconstruction,
    )

    from ctapipe_io_magic import MAGICEventSource

    params_list = [
        "width_M1",
        "width_M2",
        "length_M1",
        "length_M2",
        "size_M1",
        "size_M2",
        "delta_M1",
        "delta_M2",
        "cogx_M1",
        "cogx_M2",
        "cogy_M1",
        "cogy_M2",
        "corex",
        "corey",
        "az",
        "zd",
        "hmax",
        "impact_M1",
        "impact_M2",
        "slope_M1",
        "slope_M2",
    ]

    source = MAGICEventSource(
        input_url=dataset_calibrated_M1,
        process_run=False,
    )

    is_mc = source.is_simulation
    run_number = source.run_numbers[0]

    if is_mc:
        config_params = {
            "magic-cta-pipe-input": {
                "MCP-path": str(
                    test_data
                    / "simulated/dl1_stereo"
                    / f"dl1_stereo_magic_only_MAGIC_GA_za35to50.Run{run_number}.h5"
                )
            },
            "MARS-input": {"MARS-path": str(dataset_superstar)},
            "Params": params_list,
            "Output_paths": {
                "file_output_directory": str(test_data / "simulated/test_params"),
                "image_output_directory": str(test_data / "simulated/test_params"),
            },
        }
    else:
        config_params = {
            "magic-cta-pipe-input": {
                "MCP-path": str(
                    test_data
                    / "real/dl1_stereo"
                    / f"dl1_stereo_magic_only_MAGIC.Run0{run_number}.h5"
                )
            },
            "MARS-input": {"MARS-path": str(dataset_superstar)},
            "Params": params_list,
            "Output_paths": {
                "file_output_directory": str(test_data / "real/test_params"),
                "image_output_directory": str(test_data / "real/test_params"),
            },
        }

    config_params_file = str(tmp_path / "compare_params_config.yaml")

    with open(config_params_file, "w") as outfile:
        yaml.dump(config_params, outfile, default_flow_style=False)

    config_mcp = {
        "mc_tel_ids": {"LST-1": 1, "MAGIC-I": 2, "MAGIC-II": 3},
        "MAGIC": {
            "magic_clean": {
                "use_time": True,
                "use_sum": True,
                "picture_thresh": 6,
                "boundary_thresh": 3.5,
                "max_time_off": 4.5,
                "max_time_diff": 1.5,
                "find_hotpixels": True,
                "pedestal_type": "from_extractor_rndm",
            },  # select 'fundamental', 'from_extractor' or 'from_extractor_rndm'
        },
        "stereo_reco": {"quality_cuts": "(intensity > 50) & (width > 0)"},
    }

    config_mcp_file = str(tmp_path / "config_mcp.yaml")

    with open(config_mcp_file, "w") as outfile:
        yaml.dump(config_mcp, outfile, default_flow_style=False)

    with open(config_mcp_file, "rb") as f:
        config_mcp = yaml.safe_load(f)

    if is_mc:
        magic_calib_to_dl1(
            dataset_calibrated_M1, test_data / "simulated/dl1", config_mcp, False
        )
        magic_calib_to_dl1(
            dataset_calibrated_M2, test_data / "simulated/dl1", config_mcp, False
        )
        merge_hdf_files(
            test_data / "simulated/dl1",
            output_dir=test_data / "simulated/dl1_merged",
            run_wise=True,
            subrun_wise=False,
        )
        stereo_reconstruction(
            test_data
            / "simulated/dl1_merged"
            / f"dl1_MAGIC_GA_za35to50.Run{run_number}.h5",
            test_data / "simulated/dl1_stereo",
            config_mcp,
            magic_only_analysis=True,
        )
    else:
        magic_calib_to_dl1(
            dataset_calibrated_M1, test_data / "real/dl1", config_mcp, True
        )
        magic_calib_to_dl1(
            dataset_calibrated_M2, test_data / "real/dl1", config_mcp, True
        )
        merge_hdf_files(
            test_data / "real/dl1",
            output_dir=test_data / "real/dl1_merged",
            run_wise=True,
            subrun_wise=False,
        )
        stereo_reconstruction(
            test_data / "real/dl1_merged" / f"dl1_MAGIC.Run0{run_number}.h5",
            test_data / "real/dl1_stereo",
            config_mcp,
            magic_only_analysis=True,
        )
    list_compare_parameters, frac = compare_hillas_stereo_parameters(
        config_file=config_params_file, params_key="events/parameters", plot_image=True
    )
    print(
        "all = ",
        len(list_compare_parameters),
        " bad: ",
        sum(list_compare_parameters),
        np.array(params_list)[list_compare_parameters],
    )
    print(list_compare_parameters)
    print(frac)
    assert list_compare_parameters == [False] * len(list_compare_parameters)
