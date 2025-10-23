"""
Evaluates NSB level for a LST run (as a median over the NSB values for a subset of subruns)

One txt file per run is created here: its content is a (date,run,NSB) n-tuple and its title contain an information about the NSB-bin to which the run belongs (according to the list of NSB values provided in the config file)

Usage:
$ LSTnsb (-c MCP_config) -i run -d date -l lstchain_config (-s N_subruns)
"""
import glob
import logging
import sys

import numpy as np
import pandas as pd
import yaml
from lstchain.image.modifier import calculate_noise_parameters

from magicctapipe.io import resource_file
from magicctapipe.utils import NO_TAILCUT, auto_MCP_parser

__all__ = ["nsb"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def update_mod(mod, n_sub, denominator, index, n_noise):
    """
    Function to update the step used to extract the subruns for the NSB evaluation

    Parameters
    ----------
    mod : int
        Sampling step
    n_sub : int
        Number of subruns in the run
    denominator : int
        Number of subruns to be used to evaluate NSB for a run
    index : int
        Index of the currently used subrun
    n_noise : int
        Number of NSB values already computed

    Returns
    -------
    int
        Sampling step
    """
    if (n_sub > denominator) and (denominator > n_noise):
        mod = (n_sub - index) // (denominator - n_noise)
    return mod


def nsb(run_list, simtel, lst_config, run_number, denominator):

    """
    Here we compute the NSB value for a run based on a subset of its subruns

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
    denominator : int
        Number of subruns to be used to evaluate NSB for a run

    Returns
    -------
    list
        List of the sub-run wise NSB values
    """

    noise = []

    if len(run_list) == 0:
        logger.warning(
            "There is no subrun matching the provided run number. Check the list of the LST runs (LST_runs.txt)"
        )
        return
    if len(run_list) <= denominator:
        mod = 1
    else:
        mod = len(run_list) // denominator

    logger.info("NSB levels (sub-runs): \n")
    for ii in range(0, len(run_list)):
        subrun = run_list[ii].split(".")[-2]
        if mod == 0:
            break
        if ii % mod == 0:
            try:
                a, _, _ = calculate_noise_parameters(simtel, run_list[ii], lst_config)
                if a is not None:
                    if a > 0.0:
                        noise.append(a)
                        logger.info(a)
                    else:
                        df_subrun = pd.read_hdf(
                            run_list[ii],
                            key="dl1/event/telescope/parameters/LST_LSTCam",
                        )
                        n_ped = len(df_subrun[df_subrun["event_type"] == 2])
                        if n_ped > 0:
                            noise.append(a)
                            logger.info(a)
                        else:
                            mod = update_mod(
                                mod, len(run_list), denominator, ii, len(noise)
                            )
                            logger.warning(
                                f"NSB level could not be adequately evaluated for subrun {subrun} (missing pedestal events): skipping this subrun..."
                            )
                else:
                    mod = update_mod(mod, len(run_list), denominator, ii, len(noise))
                    logger.warning(
                        f"NSB level is None for subrun {subrun} (missing interleaved FF): skipping this subrun..."
                    )

            except IndexError:

                mod = update_mod(mod, len(run_list), denominator, ii, len(noise))
                logger.warning(
                    f"Subrun {subrun} caused an error in the NSB level evaluation for run {run_number}. Check reports before using it"
                )
    return noise


def main():

    """
    Main function
    """

    parser = auto_MCP_parser()
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
    parser.add_argument(
        "--lstchain-config",
        "-l",
        dest="lst_conf",
        type=str,
        help="lstchain configuration file",
    )
    parser.add_argument(
        "--denominator",
        "-s",
        dest="denominator",
        type=int,
        default=25,
        help="Number of subruns to be processed",
    )
    parser.add_argument(
        "--path",
        "-p",
        dest="path_lst",
        type=str,
        help="Path where LST DL1 file is located",
    )
    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    run_number = args.run
    date = args.day
    denominator = args.denominator
    lst_config = args.lst_conf
    simtel = config["expert_parameters"]["simtel_nsb"]
    nsb_list = config["expert_parameters"]["nsb"]
    width = np.diff(nsb_list, append=[nsb_list[-1] + 0.5]) / 2.0
    nsb_limit = [-0.01] + list(
        nsb_list + width
    )  # arbitrary small negative number so that 0.0 > nsb_limit[0]
    LST_files = np.sort(glob.glob(f"nsb_LST_*_{run_number}.txt"))

    if len(LST_files) == 1:
        logger.info(f"Run {run_number} already processed")
        return
    config_db = config["general"]["base_db_config_file"]
    if config_db == "":
        config_db = resource_file("database_config.yaml")

    with open(
        config_db, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict_db = yaml.safe_load(fc)

    LST_h5 = config_dict_db["database_paths"]["LST"]
    LST_key = config_dict_db["database_keys"]["LST"]
    df_LST = pd.read_hdf(
        LST_h5,
        key=LST_key,
    )

    tailcut = df_LST[df_LST.LST1_run == run_number].iloc[0]["tailcut"]
    if tailcut == "":
        logger.warning(
            f"No tailcut information in the LST database for run {run_number}. Please check directories on the cluster and database"
        )
        sys.exit(NO_TAILCUT)

    inputdir = args.path_lst
    run_list = np.sort(glob.glob(f"{inputdir}/dl1*Run*{run_number}.*.h5"))
    noise = nsb(run_list, simtel, lst_config, run_number, denominator)
    if len(noise) == 0:
        logger.warning(
            "No NSB value could be evaluated: check the observation logs (observation problems, car flashes...)"
        )
        return
    median_NSB = np.median(noise)
    logger.info("\n\n")
    logger.info(f"Run n. {run_number}, NSB median {median_NSB}")

    for j in range(0, len(nsb_list)):
        if (median_NSB <= nsb_limit[j + 1]) & (median_NSB > nsb_limit[j]):
            with open(f"nsb_LST_{nsb_list[j]}_{run_number}.txt", "a+") as f:
                f.write(f"{date},{run_number},{median_NSB}\n")
        if median_NSB > nsb_limit[-1]:
            with open(f"nsb_LST_high_{run_number}.txt", "a+") as f:
                f.write(f"{date},{run_number},{median_NSB}\n")
            break


if __name__ == "__main__":
    main()
