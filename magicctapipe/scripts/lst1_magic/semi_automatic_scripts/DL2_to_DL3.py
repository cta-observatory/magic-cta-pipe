"""
This script creates the bash scripts necessary to apply "lst1_magic_dl2_to_dl3.py"
to the DL2. It also creates new subdirectories associated with
the data level 3.

Usage:
$ python new_DL2_to_DL3.py -c configuration_file.yaml (-d list_dense.txt)
"""
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
from magicctapipe.utils import auto_MCP_parser

__all__ = ["configuration_DL3", "DL2_to_DL3"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def configuration_DL3(target_dir, source_name, config_file, ra, dec):
    """
    This function creates the configuration file needed for the DL2 to DL3 conversion

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    source_name : str
        Source name
    config_file : str
        Path to MCP configuration file (e.g., resources/config.yaml)
    ra : float
        Source RA
    dec : float
        Source Dec
    """

    if config_file == "":
        config_file = resource_file("config.yaml")

    with open(
        config_file, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)
    DL3_config = config_dict["dl2_to_dl3"]
    DL3_config["source_name"] = source_name
    DL3_config["source_ra"] = f"{ra} deg"
    DL3_config["source_dec"] = f"{dec} deg"
    conf = {
        "mc_tel_ids": config_dict["mc_tel_ids"],
        "dl2_to_dl3": DL3_config,
    }

    conf_dir = f"{target_dir}/v{__version__}/{source_name}"
    os.makedirs(conf_dir, exist_ok=True)

    file_name = f"{conf_dir}/config_DL3.yaml"

    with open(file_name, "w") as f:
        yaml.dump(conf, f, default_flow_style=False)


def DL2_to_DL3(
    target_dir,
    source,
    env_name,
    IRF_dir,
    df_LST,
    cluster,
    MC_v,
    version,
    nice,
    IRF_cuts_type,
    LST_date,
    dense_list,
):
    """
    This function creates the bash scripts to run lst1_magic_dl2_to_dl3.py on the real data.

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    source : str
        Source name
    env_name : str
        Conda enviroment name
    IRF_dir : str
        Path to the IRFs
    df_LST : :class:`pandas.DataFrame`
        Dataframe collecting the LST1 runs (produced by the create_LST_table script)
    cluster : str
        Cluster system
    MC_v : str
        Version of MC processing
    version : str
        Version of the input (stereo subruns) data
    nice : int or None
        Job priority
    IRF_cuts_type : str
        Type of IRFS to be used: global cuts (with cut value) or dynamic cuts (with efficiencies)
    LST_date : list
        List of the dates to be processed (from list_from_h5)
    dense_list : list
        List of sources that use the dense MC training line
    """
    if cluster != "SLURM":
        logger.warning(
            "Automatic processing not implemented for the cluster indicated in the config file"
        )
        return

    # Loop over all nights

    DL3_Nights = np.sort(glob.glob(f"{target_dir}/v{__version__}/{source}/DL3/*"))
    for dl3date in DL3_Nights:
        night = dl3date.split("/")[-1]
        outdir = f"{dl3date}/logs"
        File_list = np.sort(glob.glob(f"{outdir}/*.txt"))
        for file in File_list:

            if str(night) not in LST_date:
                continue
            with open(file, "r") as f:
                runs = f.readlines()
                process_size = len(runs) - 1
                run_new = []
                for run in runs:
                    wobble_offset = df_LST[df_LST.LST1_run == run].iloc[0]["wobble_offset"]
                    if str(wobble_offset) != "['0.40']":
                        print(f"wobble offset is not (or not always) 0.40 for {source}, run {run}")
                        continue
                    single_run_new = (
                        "/".join(run.split("/")[:-6])
                        + f"/v{version}/"
                        + run.split("/")[-5]
                        + "/DL2/"
                        + run.split("/")[-2]
                        + "/"
                        + run.split("/")[-1].replace("dl1_stereo", "dl2")
                    )
                    run_new.append(single_run_new)
                with open(file, "w") as g:
                    g.writelines(run_new)

            nsb = file.split("/")[-1].split("_")[1]
            period = file.split("/")[-1].split("_")[0]
            dec = df_LST[df_LST.source == source].iloc[0]["MC_dec"]
            if np.isnan(dec):
                print(f"MC_dec is NaN for {source}")
                continue
            dec = str(dec).replace(".", "").replace("-", "min_")
            IRFdir = f"{IRF_dir}/{period}/NSB{nsb}/GammaTest/v{MC_v}/{IRF_cuts_type}/dec_{dec}{'_high_density' if source in dense_list else ''}/"
            if (not os.path.isdir(IRFdir)) or (
                len(glob.glob(f"{IRFdir}/irf_*fits.gz")) < 1
            ):
                print(f"no IRF availables in {IRFdir}")
                continue

            slurm = slurm_lines(
                queue="short",
                job_name=f"{source}_DL2_to_DL3",
                nice_parameter=nice,
                array=process_size,
                mem="50g",
                out_name=f"{outdir}/slurm-%x.%A_%a",
            )
            rc = rc_lines(
                store="$SAMPLE ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID}",
                out=f"{outdir}/list_{nsb}_{period}_{night}",
            )
            out_file = outdir.rstrip("/logs")

            lines = (
                slurm
                + [
                    f"SAMPLE_LIST=($(<{file}))\n",
                    "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n",
                    f"export LOG={outdir}",
                    "/DL2_to_DL3_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log\n",
                    f"conda run -n {env_name} lst1_magic_dl2_to_dl3 --input-file-dl2 $SAMPLE --input-dir-irf {IRFdir} --output-dir {out_file} --config-file {target_dir}/v{__version__}/{source}/config_DL3.yaml >$LOG 2>&1\n\n",
                ]
                + rc
            )

            with open(f"{source}_DL2_to_DL3_{nsb}_{period}_{night}.sh", "w") as f:
                f.writelines(lines)


def main():
    """
    Here we read the config_auto_MCP.yaml file and call the functions defined above.
    """

    parser = auto_MCP_parser()
    parser.add_argument(
        "--dense_MC_sources",
        "-d",
        dest="dense_list",
        type=str,
        help="File with name of sources to be processed with the dense MC train line",
    )

    args = parser.parse_args()
    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)

    dense_list = []
    if args.dense_list is not None:
        with open(args.dense_list) as d:
            dense_list = d.read().splitlines()

    target_dir = Path(config["directories"]["workspace_dir"])
    IRF_dir = config["directories"]["IRF"]

    source_in = config["data_selection"]["source_name_database"]
    source = config["data_selection"]["source_name_output"]
    env_name = config["general"]["env_name"]
    config_file = config["general"]["base_config_file"]
    cluster = config["general"]["cluster"]
    in_version = config["directories"]["real_input_version"]
    if in_version == "":
        in_version = __version__
    nice_parameter = config["general"]["nice"] if "nice" in config["general"] else None
    MC_v = config["directories"]["MC_version"]
    if MC_v == "":
        MC_v = __version__
    IRF_cuts_type = config["general"]["IRF_cuts_type"]
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
        # cp the .txt files from DL1 stereo anaysis to be used again.
        DL2_Nights = np.sort(
            glob.glob(f"{target_dir}/v{in_version}/{source_name}/DL2/*")
        )
        LST_runs_and_dates = f"{source_name}_LST_runs.txt"
        LST_date = []
        for i in np.genfromtxt(LST_runs_and_dates, dtype=str, delimiter=",", ndmin=2):
            LST_date.append(str(i[0].replace("_", "")))
        LST_date = list(set(LST_date))

        for night in DL2_Nights:
            nightdate = night.split("/")[-1]
            if nightdate in LST_date:
                outdir = (
                    f"{target_dir}/v{__version__}/{source_name}/DL3/{nightdate}/logs"
                )
                os.makedirs(outdir, exist_ok=True)
                File_list = glob.glob(f"{night}/logs/*.txt")
                for file in File_list:
                    os.system(f"cp {file} {outdir}")

        ra = df_LST[df_LST.source == source_name].iloc[0]["ra"]
        dec = df_LST[df_LST.source == source_name].iloc[0]["dec"]
        if np.isnan(dec) or np.isnan(ra):
            print(f"source Ra and/or Dec is NaN for {source_name}")
            continue
        print("***** Generating file config_DL3.yaml...")
        print(
            f"***** This file can be found in {target_dir}/v{__version__}/{source_name}"
        )
        configuration_DL3(target_dir, source_name, config_file, ra, dec)

        print("***** Generating bash scripts...")
        DL2_to_DL3(
            target_dir,
            source_name,
            env_name,
            IRF_dir,
            df_LST,
            cluster,
            MC_v,
            in_version,
            nice_parameter,
            IRF_cuts_type,
            LST_date,
            dense_list,
        )
        list_of_dl3_scripts = np.sort(glob.glob(f"{source_name}_DL2_to_DL3*.sh"))
        if len(list_of_dl3_scripts) < 1:
            logger.warning(f"No bash scripts for {source_name}")
            continue
        launch_jobs = ""
        for n, run in enumerate(list_of_dl3_scripts):
            launch_jobs += (" && " if n > 0 else "") + f"sbatch {run}"
        os.system(launch_jobs)


if __name__ == "__main__":
    main()
