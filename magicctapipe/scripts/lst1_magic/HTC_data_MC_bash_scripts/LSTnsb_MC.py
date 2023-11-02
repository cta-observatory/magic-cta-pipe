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

__all__=['nsb']

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def nsb(run_list, simtel, lst_config, run_number):
    """Here we compute the NSB value for a run
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
    if len(run_list) == 0:
        return
    if len(run_list) < 25:
        mod = 1
    else:
        mod = int(len(run_list) / 25)
    for ii in range(0, len(run_list)):
        if mod == 0:
            break
        if ii % mod == 0:
            try:
                a, _, _ = calculate_noise_parameters(simtel, run_list[ii], lst_config)
                noise.append(a)
            except IndexError:
                mod = mod - 1
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
    simtel = "/storage/gpfs_data/ctalocal/LST1/jointMAGIC/Crab/mc/GammaDiffuse/node_simtel_theta_37.661_az_270.641_/simtel_theta_37.661_az_270.641_run10.simtel.gz"

    source = config["directories"]["target_name"]
    logger.info(run)
    lst_config = "/storage/gpfs_data/ctalocal/evisentin/magic-cta-pipe/magicctapipe/scripts/lst1_magic/HTC_data_MC_bash_scripts/lstchain_standard_config.json"
    run_number = run.split(",")[1]
    LST_files = np.sort(glob.glob(f"{source}_LST_{run_number}.txt"))

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
    inputdir = f"/storage/gpfs_data/ctalocal/evisentin/LST_data"
    run_list = np.sort(glob.glob(f"{inputdir}/dl1*Run*{run_number}.*.h5"))
    logger.info(run_list)
    noise=nsb(run_list, simtel, lst_config, run_number)
    if len(noise)==0:
        return
    a=np.median(noise)
    logger.info(f"Run n. {run_number}, nsb median {a}")
    
    with open(f"{source}_LST_{run_number}.txt", "a+") as f:
        f.write(f"{a}\n")


if __name__ == "__main__":
    main()
