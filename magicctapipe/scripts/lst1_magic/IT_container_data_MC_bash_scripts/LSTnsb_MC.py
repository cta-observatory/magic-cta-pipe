"""
Evaluates NSB level for a LST run
"""
import argparse
import glob
import logging
import os

import numpy as np
import yaml
from lstchain.image.modifier import calculate_noise_parameters

__all__ = ["nsb"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def nsb(run_list, simtel, lst_config, run_number):

    """
    Here we compute the NSB value for a run based on a subset of subruns.

    Parameters
    ----------
    run_list : list
        List of subruns in the run
    simtel : str
        Simtel (MC) file to be used to evaluate the extra noise in dim pixels
    lst_config : str
        LST configuration file (cf. lstchain)
    run_number : int
        LST run number

    Returns
    -------
    list
        List of the sub-run wise NSB values
    """

    noise = []
    denominator = 25
    if len(run_list) == 0:
        logger.warning(
            "There is no subrun matching the provided run number. Check the list of the LST runs (LST_runs.txt)"
        )
        return
    if len(run_list) < denominator:
        mod = 1
    else:
        mod = int(len(run_list) / denominator)
    for ii in range(0, len(run_list)):
        subrun = run_list[ii].split(".")[-2]
        if mod == 0:
            break
        if ii % mod == 0:
            try:
                a, _, _ = calculate_noise_parameters(simtel, run_list[ii], lst_config)
                noise.append(a)
            except IndexError:
                mod = int(len(run_list) / (denominator + 1))
                logger.warning(
                    f"Subrun {subrun} caused an error in the NSB level evaluation for run {run_number}. Check reports before using it"
                )
    return noise


def main():

    """
    Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config_general.yaml",
        help="Path to a configuration file",
    )
    parser.add_argument(
        "--input-run",
        "-i",
        dest="run",
        type=str,
        help="Run to be processed",
    )
    parser.add_argument(
        "--day",
        "-d",
        dest="day",
        type=str,
        help="Day of the run to be processed",
    )

    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)

    run_number = args.run
    date = args.day
    simtel = "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/sim_telarray/node_theta_14.984_az_355.158_/output_v1.4/simtel_corsika_theta_14.984_az_355.158_run10.simtel.gz"
    source = config["directories"]["target_name"]
    lst_version = config["general"]["LST_version"]
    lst_tailcut = config["general"]["LST_tailcut"]
    lst_config = "lstchain_standard_config.json"
    LST_files = np.sort(glob.glob(f"{source}_LST_nsb_*{run_number}*.txt"))

    if len(LST_files) > 1:
        logger.warning(
            f"More than one files exists for run {run_number}. Removing all these files and evaluating it again."
        )
        for repeated_files in LST_files:
            os.remove(repeated_files)
        LST_files = []
    elif len(LST_files) == 1:
        logger.info(f"Run {run_number} already processed.")
        return

    date_lst = date.split("_")[0] + date.split("_")[1] + date.split("_")[2]
    inputdir = f"/fefs/aswg/data/real/DL1/{date_lst}/{lst_version}/{lst_tailcut}"
    run_list = np.sort(glob.glob(f"{inputdir}/dl1*Run*{run_number}.*.h5"))
    noise = nsb(run_list, simtel, lst_config, run_number)
    if len(noise) == 0:
        logger.warning(
            "No NSB value could be evaluated: check the observation logs (observation problems, car flashes...)"
        )
        return
    a = np.median(noise)
    logger.info(f"Run n. {run_number}, nsb median {a}")

    with open(f"{source}_LST_nsb_{run_number}.txt", "a+") as f:
        f.write(f"{a}\n")


if __name__ == "__main__":
    main()
