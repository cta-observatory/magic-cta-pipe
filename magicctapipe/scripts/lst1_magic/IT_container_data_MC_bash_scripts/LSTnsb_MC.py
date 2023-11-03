"""
Evaluates NSB level for a LST run
"""
import argparse
import glob
import logging
import numpy as np
import os
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
    run_list: list
        List of subruns in the run
    simtel: str
        Simtel (MC) file to be used to evaluate the extra noise in dim pixels
    lst_config: str
        LST configuration file (cf. lstchain)
    run number: int
        LST run number

    """
    noise = []
    denominator = 25
    if len(run_list) == 0:
        return
    if len(run_list) < denominator:
        mod = 1
    else:
        mod = int(len(run_list) / denominator)
    for ii in range(0, len(run_list)):
        if mod == 0:
            break
        if ii % mod == 0:
            try:
                a, _, _ = calculate_noise_parameters(simtel, run_list[ii], lst_config)
                noise.append(a)
            except IndexError:
                mod = int(len(run_list) / (denominator + 1))
                logger.info(
                    f"WARNING: a subrun caused an error in the NSB level evaluation for run {run_number}. Check reports before using it"
                )
    return noise


def main():
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

    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)

    run = args.run
    simtel = "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/sim_telarray/node_theta_14.984_az_355.158_/output_v1.4/simtel_corsika_theta_14.984_az_355.158_run10.simtel.gz"

    source = config["directories"]["target_name"]

    lst_config = "lstchain_standard_config.json"
    run_number = run.split(",")[1]
    LST_files = np.sort(glob.glob(f"{source}_LST_nsb_*{run_number}*.txt"))

    date = run.split(",")[0]
    if len(LST_files) > 1:
        logger.info(
            f"run {run_number} classified in more than one NSB bin. Removing all these files and evaluating it again."
        )
        for repeated_files in LST_files:
            os.remove(repeated_files)
        LST_files = []
    elif len(LST_files) == 1:
        logger.info(f"run {run_number} already processed.")
        return

    date_lst = date.split("_")[0] + date.split("_")[1] + date.split("_")[2]
    inputdir = f"/fefs/aswg/data/real/DL1/{date_lst}/v0.9/tailcut84"
    run_list = np.sort(glob.glob(f"{inputdir}/dl1*Run*{run_number}.*.h5"))
    noise = nsb(run_list, simtel, lst_config, run_number)
    if len(noise) == 0:
        return
    a = np.median(noise)
    logger.info(f"Run n. {run_number}, nsb median {a}")

    with open(f"{source}_LST_nsb_{run_number}.txt", "a+") as f:
        f.write(f"{a}\n")


if __name__ == "__main__":
    main()
