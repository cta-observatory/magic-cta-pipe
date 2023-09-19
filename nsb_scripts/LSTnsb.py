"""
Evaluates NSB level for a LST run
"""
import os
import argparse
from lstchain.image.modifier import calculate_noise_parameters
import numpy as np
import yaml
import glob
import logging


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


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

    nsb = config["general"]["nsb"]
    width = [a / 2 - b / 2 for a, b in zip(nsb[1:], nsb[:-1])]
    source = config["directories"]["target_name"]
    width.append(0.25)
    nsb_limit = [a + b for a, b in zip(nsb[:], width[:])]
    nsb_limit.insert(0, 0)

    lst_config = "lstchain_standard_config.json"
    run_number = run.split(",")[1]
    LST_files = np.sort(glob.glob(f"{source}_LST_[0-9]*_{run_number}.txt"))

    date = run.split(",")[0]
    if len(LST_files) > 1:
        logger.info(
            f"run {run_number} classified in more than one NSB bin. Removing all these files and evaluating it again"
        )
        for kk in LST_files:
            os.remove(kk)
        LST_files = []
    if len(LST_files) == 1:
        logger.info(f"run {run_number} already processed")
        return

    date_lst = date.split("_")[0] + date.split("_")[1] + date.split("_")[2]
    inputdir = f"/fefs/aswg/data/real/DL1/{date_lst}/v0.9/tailcut84"
    run_list = np.sort(glob.glob(f"{inputdir}/dl1*Run*{run_number}.*.h5"))
    noise = []
    if len(run_list) == 0:
        return
    if len(run_list) < 25:
        mod = 1
    else:
        mod = int(len(run_list) / 25)
    for ii in range(0, len(run_list)):
        if ii % mod == 0:
            a, b, c = calculate_noise_parameters(simtel, run_list[ii], lst_config)
            noise.append(a)
    a = sum(noise) / len(noise)
    std = np.std(noise)
    logger.info(f"Run n. {run_number}, nsb average (all) {a}, std {std}")
    subrun_ok = []
    for sr in range(0, len(noise)):
        if np.abs(noise[sr] - a) < 3 * std:
            subrun_ok.append(noise[sr])
    a = sum(subrun_ok) / len(subrun_ok)
    logger.info(f"Run n. {run_number}, nsb average (w/o outliers) {a}")
    for j in range(0, len(nsb)):
        if (a < nsb_limit[j + 1]) & (a > nsb_limit[j]):
            with open(f"{source}_LST_{nsb[j]}_{run_number}.txt", "a+") as f:
                f.write(run + "\n")


if __name__ == "__main__":
    main()