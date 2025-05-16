"""
This script creates the bash scripts necessary to apply "lst1_magic_dl1_stereo_to_dl2.py"
to the DL1 stereo data. It also creates new subdirectories associated with
the data level 2.

Usage:
$ DL1_to_DL2 -c configuration_file.yaml
"""
import argparse
import datetime
import glob
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from magicctapipe import __version__
from magicctapipe.io import resource_file
from magicctapipe.scripts.lst1_magic.semi_automatic_scripts.clusters import (
    rc_lines,
    slurm_lines,
)

__all__ = ["ST_NSB_List", "bash_DL1Stereo_to_DL2"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def ST_NSB_List(target_dir, nsb_list, source, df_LST, MAGIC_obs_periods, version):
    """
    This function creates the lists of runs separeted by run period and NSB level.

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    nsb_list : list
        List of the MC NSB values
    source : str
        Source name
    df_LST : :class:`pandas.DataFrame`
        Dataframe collecting the LST1 runs (produced by the create_LST_table script)
    MAGIC_obs_periods : dict
        Dictionary of MAGIC observation periods (key = name of period, value = list of begin/end dates)
    version : str
        Version of the input (stereo subruns) data
    """
    width = np.diff(nsb_list, append=[nsb_list[-1] + 0.5]) / 2.0
    nsb_limit = [-0.01] + list(
        nsb_list + width
    )  # arbitrary small negative number so that 0.0 > nsb_limit[0]

    # Loops over all runs of all nights
    Nights_list = np.sort(
        glob.glob(f"{target_dir}/v{version}/{source}/DL1Stereo/Merged/*")
    )
    for night in Nights_list:
        # Night period

        night_date = night.split("/")[-1]
        outdir = f"{target_dir}/v{__version__}/{source}/DL2/{night_date}/logs"
        os.makedirs(outdir, exist_ok=True)

        date_magic = datetime.datetime.strptime(
            night_date, "%Y%m%d"
        ) + datetime.timedelta(days=1)
        period = None
        for p_name, date_list in MAGIC_obs_periods.items():
            for date1, date2 in date_list:
                date_init = datetime.datetime.strptime(date1, "%Y_%m_%d")
                date_end = datetime.datetime.strptime(date2, "%Y_%m_%d")
                if (date_magic >= date_init) and (date_magic <= date_end):
                    period = p_name

        if period is None:
            print(f"Could not identify MAGIC period for LST night {night_date}")
            continue

        Run_list = glob.glob(f"{night}/*.h5")
        for Run in Run_list:
            # getting the run NSB
            run_str = Run.split("/")[-1].split(".")[1]
            run_LST_id = run_str.lstrip("Run")
            nsb = df_LST[df_LST["LST1_run"] == run_LST_id]["nsb"].tolist()[0]
            # rounding the NSB to the nearest MC nsb value
            for j in range(0, len(nsb_list)):
                if (nsb <= nsb_limit[j + 1]) & (nsb > nsb_limit[j]):
                    nsb = nsb_list[j]

            # Writing on output .txt file
            if nsb <= nsb_limit[-1]:
                with open(
                    f"{outdir}/{period}_{nsb}_{night_date}.txt",
                    "a+",
                ) as file:
                    file.write(f"{Run}\n")


def bash_DL1Stereo_to_DL2(
    target_dir, source, env_name, cluster, RF_dir, df_LST, MC_v, version, nice
):
    """
    This function generates the bashscript for running the DL1Stereo to DL2 analisys.

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    source : str
        Source name
    env_name : str
        Conda enviroment name
    cluster : str
        Cluster system
    RF_dir : str
        Path to the RFs
    df_LST : :class:`pandas.DataFrame`
        Dataframe collecting the LST1 runs (produced by the create_LST_table script)
    MC_v : str
        Version of MC processing
    version : str
        Version of the input (stereo subruns) data
    nice : int or None
        Job priority
    """
    if cluster != "SLURM":
        logger.warning(
            "Automatic processing not implemented for the cluster indicated in the config file"
        )
        return

    process_name = source
    LST_runs_and_dates = f"{source}_LST_runs.txt"
    LST_date = []
    for i in np.genfromtxt(LST_runs_and_dates, dtype=str, delimiter=",", ndmin=2):
        LST_date.append(str(i[0].replace("_", "")))
    LST_date = list(set(LST_date))
    Nights_list = np.sort(
        glob.glob(f"{target_dir}/v{version}/{source}/DL1Stereo/Merged/*")
    )

    for night in Nights_list:
        night_date = night.split("/")[-1]
        outdir = f"{target_dir}/v{__version__}/{source}/DL2/{night_date}/logs"
        File_list = glob.glob(f"{outdir}/ST*.txt")
        night_date = night.split("/")[-1]
        if str(night_date) not in LST_date:
            continue

        for file in File_list:
            with open(file, "r") as f:
                process_size = len(f.readlines()) - 1
            if process_size < 0:
                continue
            nsb = file.split("/")[-1].split("_")[1]
            period = file.split("/")[-1].split("_")[0]
            dec = df_LST[df_LST.source == source].iloc[0]["MC_dec"]
            if np.isnan(dec):
                print(f"MC_dec is NaN for {source}")
                continue
            dec = str(dec).replace(".", "").replace("-", "min_")

            RFdir = f"{RF_dir}/{period}/NSB{nsb}/v{MC_v}/dec_{dec}/"
            if (not os.path.isdir(RFdir)) or (len(glob.glob(f"{RFdir}/*joblib")) < 3):
                print(f"no RF availables in {RFdir}")
                continue
            rfsize = 0
            for rffile in glob.glob(f"{RFdir}/disp*joblib"):
                rfsize = rfsize + os.path.getsize(rffile) / (1024 * 1024 * 1024)
            rfsize = (rfsize * 1.75) + 2
            slurm = slurm_lines(
                queue="short",
                job_name=f"{process_name}_DL1_to_DL2",
                nice_parameter=nice,
                array=process_size,
                mem=f"{int(rfsize)}g",
                out_name=f"{outdir}/slurm-%x.%A_%a",
            )
            rc = rc_lines(
                store="$SAMPLE ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID}",
                out=f"{outdir}/list_{nsb}_{period}",
            )
            out_file = outdir.rstrip("/logs")
            lines = (
                slurm
                + [
                    f"SAMPLE_LIST=($(<{file}))\n",
                    "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n",
                    f"export LOG={outdir}",
                    "/DL1_to_DL2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log\n",
                    f"conda run -n {env_name} lst1_magic_dl1_stereo_to_dl2 --input-file-dl1 $SAMPLE --input-dir-rfs {RFdir} --output-dir {out_file} >$LOG 2>&1\n\n",
                ]
                + rc
            )
            with open(
                f'{source}_DL1_to_DL2_{file.split("/")[-1].rstrip("txt")}sh',
                "w",
            ) as f:
                f.writelines(lines)


def main():
    """
    Here we read the config_auto_MCP.yaml file and call the functions defined above.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config_auto_MCP.yaml",
        help="Path to a configuration file",
    )

    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)

    target_dir = Path(config["directories"]["workspace_dir"])
    RF_dir = config["directories"]["RF"]
    env_name = config["general"]["env_name"]
    MAGIC_obs_periods = config["expert_parameters"]["MAGIC_obs_periods"]
    nsb_list = config["expert_parameters"]["nsb"]

    source_in = config["data_selection"]["source_name_database"]
    source = config["data_selection"]["source_name_output"]
    MC_v = config["directories"]["MC_version"]
    if MC_v == "":
        MC_v = __version__

    cluster = config["general"]["cluster"]
    in_version = config["directories"]["real_input_version"]
    if in_version == "":
        in_version = __version__
    nice_parameter = config["general"]["nice"] if "nice" in config["general"] else None

    # LST dataframe
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

    if source_in is None:
        source_list = joblib.load("list_sources.dat")
    else:
        source_list = [source]
    for source_name in source_list:
        ST_NSB_List(
            target_dir,
            nsb_list,
            source_name,
            df_LST,
            MAGIC_obs_periods,
            in_version,
        )

        bash_DL1Stereo_to_DL2(
            target_dir,
            source_name,
            env_name,
            cluster,
            RF_dir,
            df_LST,
            MC_v,
            in_version,
            nice_parameter,
        )
        list_of_dl2_scripts = np.sort(glob.glob(f"{source_name}_DL1_to_DL2*.sh"))
        if len(list_of_dl2_scripts) < 1:
            logger.warning(f"No bash scripts for {source_name}")
            continue
        launch_jobs = ""
        for n, run in enumerate(list_of_dl2_scripts):
            launch_jobs += (" && " if n > 0 else "") + f"sbatch {run}"
        os.system(launch_jobs)


if __name__ == "__main__":
    main()
