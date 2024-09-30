#!/usr/bin/env python
# coding: utf-8

"""
This script search the time offset for using MAGIC data which the GPS is broken.
The offset time in such a case is large (few second) and changed gradually (few us/sec). The drift speed is known as constant in each subrun.
In this script we use the subrun files of MAGIC and LST as input files, and make a combination for each event timestamp. From each combination, the best matched time will be found.
It will take few tens of minutes to make each npy files. Basically, each MAGIC run contains ~10 subruns and so this step take few hours to be finished.

Usage:
$ python lst1_magic_event_coincidence.py data

If you have some runs the input data directory should be separated for each run and use them to save the time, as like,
data1/LST/
dl1_LST-1.Run17220.0000.h5 dl1_LST-1.Run17220.0001.h5 ...
data1/MAGIC/
dl1_MAGIC.Run05114133.001.h5 dl1_MAGIC.Run05114133.002.h5 ...

data2/LST/
dl1_LST-1.Run17221.0000.h5 dl1_LST-1.Run17221.0001.h5 ...
data2/MAGIC/
dl1_MAGIC.Run05114134.001.h5 dl1_MAGIC.Run05114134.002.h5 ...

$ python lst1_magic_event_coincidence.py data1
In parallel,
$ python lst1_magic_event_coincidence.py data2
...
"""

import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from magicctapipe.io import find_offset

file_dl1_dir = sys.argv[1]
outdir = "time_offset"
config = "../../config_dyn.yaml"

magic_run = int(
    pd.read_hdf(
        glob.glob(file_dl1_dir + "/MAGIC/dl1_MAGIC.*Run*h5")[0], key="events/parameters"
    )["obs_id"].mean()
)
lst_run = int(
    pd.read_hdf(
        glob.glob(file_dl1_dir + "/LST1/dl1_LST-1.Run*h5")[0],
        key="/dl1/event/telescope/parameters/LST_LSTCam",
    )["obs_id"].mean()
)
magic_run, lst_run = str(magic_run).zfill(8), str(lst_run).zfill(5)

i = 0
for lst_subrun_file in sorted(glob.glob(file_dl1_dir + "/LST1/dl1_LST*h5")):
    df_lst_subrun_file = pd.read_hdf(
        lst_subrun_file, key="/dl1/event/telescope/parameters/LST_LSTCam"
    )
    df_lst_subrun_file = df_lst_subrun_file.query("intensity>100")
    df_lst_subrun_file = df_lst_subrun_file[["trigger_time", "event_type"]]
    if i == 0:
        df_lst_subrun_file_2 = df_lst_subrun_file
    else:
        df_lst_subrun_file_2 = pd.concat([df_lst_subrun_file_2, df_lst_subrun_file])
    i = i + 1

try:
    os.makedirs(outdir, exist_ok=True) 
except FileExistsError:
    pass

# Find the time offset for each subrun combination
for magic_subrun_file in sorted(
    glob.glob(file_dl1_dir + "/MAGIC/dl1_MAGIC.Run" + magic_run + ".*.h5")
):
    data_magic = pd.read_hdf(magic_subrun_file, key="events/parameters")
    data_magic = data_magic.query("intensity>100")
    data_magic["trigger_time"] = (
        data_magic["time_sec"] + data_magic["time_nanosec"] * 1e-9
    )
    tel_id_m1 = config["mc_tel_ids"]["MAGIC-I"]
    tel_id_m2 = config["mc_tel_ids"]["MAGIC-II"]
    # JS: better to read the tel_ids from the config file
    data_magic_m1 = data_magic.query("tel_id==@tel_id_m1")
    data_magic_m2 = data_magic.query("tel_id==@tel_id_m2")
    magic_subrun_base = magic_subrun_file.split("/")[-1].split(".h5")[0]
    # JS: you use data_magic_m[12] only here, so you can add the query and reset index 
    # lines from before the for loop inside the loop over the telescopes here.
    data_lst = df_lst_subrun_file_2

    for M1_M2 in ["M1", "M2"]:
        if M1_M2 == "M1":
            data_magic = data_magic_m1
        else:
            data_magic = data_magic_m2
        # JS: you just reset the index before the for loop, do you need to do it again?
        data_magic.reset_index(inplace=True,drop=True)
        min_t_magic, max_t_magic = min(data_magic["trigger_time"]), max(
            data_magic["trigger_time"]
        )
        min_t_lst, max_t_lst = min(data_lst["trigger_time"]), max(
            data_lst["trigger_time"]
        )

        if min_t_magic < min_t_lst:
            if max_t_magic < max_t_lst:
                data_magic = data_magic.query(
                    "@min_t_lst < trigger_time < @max_t_magic"
                )
            else:
                data_magic = data_magic.query("@min_t_lst < trigger_time < @max_t_lst")
                data_magic.reset_index(inplace=True)#,drop=True)
                # JS: no condition needed for min_t_lst < min_t_magic?

        if len(data_magic) > 0:
            N_begin = 0
            N_final = data_magic.index[-1] - data_magic.index[0]

            N_start_ = N_begin
            N_end_ = N_start_ + 15

            print(N_start_, N_end_)
            print(data_magic[N_start_:N_end_]) 
            # This output file will be used for the next detailed offset search
            outfile = (
                outdir + "/" + magic_subrun_base + "_" + str(N_start_) + "_init.npy"
            )
            t_magic_all, N_start_out, time_offset_best, n_coincident = find_offset(
                data_magic, data_lst, N_start=N_start_, N_end=N_end_
            )

            if n_coincident != 0:
                np.save(
                    outfile.replace("MAGIC", M1_M2),
                    np.array(
                        [
                            np.mean(t_magic_all),
                            time_offset_best,
                            # JS: why is best offset saved twice in the same array ?
                            # (the same also in a similar code below))
                            time_offset_best,
                            n_coincident,
                        ]
                    ),
                )
                time_offset_center = np.load(outfile.replace("MAGIC", M1_M2))[2]
                N_end_of_run = N_final
                print(
                    "Simultaneous MAGIC+LST1 obs from MAGIC evt index",
                    N_start_,
                    "to",
                    N_end_of_run,
                )

                # This output file will be loaded by the lst_magic_event_coincidence.py
                outfile2 = (
                    outdir
                    + "/"
                    + magic_subrun_base
                    + "_"
                    + str(N_start_)
                    + "_"
                    + str(N_end_of_run)
                    + "_detail.npy"
                )
                (
                    t_magic_all,
                    time_offset_best,
                    # JS: you assign two different outputs of find_offset to the same variable 
                    # time_offset_best. If you do not need one of the function outputs you can 
                    # assign it to _
                    _,
                    n_coincident,
                ) = find_offset(
                    data_magic,
                    data_lst,
                    N_start=N_start_,
                    N_end=N_end_of_run,
                    initial_time_offset=time_offset_center,
                )
                # JS: saving is the same as with output, just the name of the variable is 
                # different, but maybe you can unify it and save after the if statement 
                # without code repetition
                np.save(
                    outfile2.replace("MAGIC", M1_M2),
                    np.array(
                        [t_magic_all, time_offset_best, time_offset_best, n_coincident]
                    ),
                )

# Create the coincident dl1 files
magic_dir_name = file_dl1_dir + "/MAGIC/"
lst_dir_name = file_dl1_dir + "/LST1/"

magic_t = []
lst_t = []
start_l, stop_l = [], []
run_l = []
obs_id_l = []
start_m, stop_m = [], []
run_m = []
obs_id_m = []

magic_dataset = glob.glob(magic_dir_name + "/*h5")
magic_dataset = sorted(magic_dataset)

for magic_data in magic_dataset:
    data_magic = pd.read_hdf(magic_data, key="events/parameters")
    data_magic["trigger_time"] = (
        data_magic["time_sec"] + data_magic["time_nanosec"] * 1e-9
    )
    start_m.append(min(data_magic["trigger_time"]))
    stop_m.append(max(data_magic["trigger_time"]))
    run_m.append(int(data_magic["obs_id"].mean()))

df_magic = pd.DataFrame({"start": start_m, "stop": stop_m, "run": run_m})
df_magic = df_magic.groupby("run").agg({"start": "min", "stop": "max"}).reset_index()#drop=True)

print(df_magic)

lst_dataset = glob.glob(lst_dir_name + "/*h5")
lst_dataset = sorted(lst_dataset)

for lst_data in lst_dataset:
    data_lst = pd.read_hdf(lst_data, key="/dl1/event/telescope/parameters/LST_LSTCam")
    start_l.append(min(data_lst["trigger_time"]))
    stop_l.append(max(data_lst["trigger_time"]))
    run_l.append(int(data_lst["obs_id"].mean()))

df_lst = pd.DataFrame({"start": start_l, "stop": stop_l, "run": run_l})
df_lst = df_lst.groupby("run").agg({"start": "min", "stop": "max"}).reset_index()#drop=True)

print(df_lst)


def print_gti(df1, df2):
    """
    This function searches the correponting subrun files between LST and MAGIC.

    Parameters
    ----------
    df1, df2 : str
        Input files which contain the timestamp and run/subrun name of MAGIC and LST.

    Returns
    -------
    array
        The array which have a run/subrun ids corresponding the simultaneous observation for MAGIC and LST.
    """
    start2_list, stop2_list = (
        df2["start"],
        df2["stop"],
    )
    i = 0
    subrun_1, subrun_2 = [], []
    for run_ in df1["run"].values:
        df1_ = df1.query("run==@run_")
        start1, stop1 = df1_["start"], df1_["stop"]
        start1, stop1 = float(start1.iloc[0]), float(stop1.iloc[0])
        j = 0
        for start2, stop2 in zip(start2_list, stop2_list):
            obs_time_magic = stop1 - start1
            obs_time_lst = stop2 - start2
            if [obs_time_magic > obs_time_lst]:
                s1, e1, s2, e2 = start1, stop1, start2, stop2
            else:
                s1, e1, s2, e2 = start2, stop2, start1, stop1
                # JS: the two cases seem to have the same commands, 
                # you could make a single if with ((..) & (...)) | ((..) & (...)) condition
            #if (s1 < s2) & (s2 < e1) == True:
            if ((s1 < s2) & (s2 < e1)) | ((s1 < e2) & (e2 < e1)):
                subrun_1.append(df1_["run"].values[0])
                subrun_2.append(df2["run"].values[j])
            #elif (s1 < e2) & (e2 < e1) == True:
            #    subrun_1.append(df1_["run"].values[0])
            #    subrun_2.append(df2["run"].values[j])
            j = j + 1
        i = i + 1
    return subrun_1, subrun_2


magic_run, lst_run = print_gti(df_magic, df_lst)
df = pd.DataFrame({"magic_run": magic_run, "lst_run": lst_run})

try:
    os.makedirs(outdir, exist_ok=True) 
except FileExistsError:
    pass

print("-- search subrun coincidence --")
for i in range(0, len(df)):
    lst_RunID = str(df["lst_run"][i]).zfill(5)
    magic_RunID = str(df["magic_run"][i]).zfill(8)

    lst_dataset_group = [L for L in lst_dataset if lst_RunID in L]
    magic_dataset_group = [m for m in magic_dataset if magic_RunID in m]

    start_l_subrun, stop_l_subrun = [], []
    subrun_l = []

    for input_file in lst_dataset_group:
        data_lst = pd.read_hdf(
            input_file, key="/dl1/event/telescope/parameters/LST_LSTCam"
        )
        start_l_subrun.append(min(data_lst["trigger_time"]))
        stop_l_subrun.append(max(data_lst["trigger_time"]))
        subrun = int(input_file.split(".")[-2])
        subrun_l.append(subrun)
    df_lst_subrun = pd.DataFrame(
        {"start": start_l_subrun, "stop": stop_l_subrun, "run": subrun_l}
    )

    start_m_subrun, stop_m_subrun = [], []
    subrun_m = []

    for input_file in magic_dataset_group:
        data_magic = pd.read_hdf(input_file, key="events/parameters")
        data_magic["trigger_time"] = (
            data_magic["time_sec"] + data_magic["time_nanosec"] * 1e-9
        )
        start_m_subrun.append(min(data_magic["trigger_time"]))
        stop_m_subrun.append(max(data_magic["trigger_time"]))
        subrun = int(input_file.split(".")[-2])
        subrun_m.append(subrun)
    df_magic_subrun = pd.DataFrame(
        {"start": start_m_subrun, "stop": stop_m_subrun, "run": subrun_m}
    )

    # Find the corresponding subrun for each file
    magic_subrun, lst_subrun = print_gti(df_magic_subrun, df_lst_subrun)
    df_subrun = pd.DataFrame({"magic_subrun": magic_subrun, "lst_subrun": lst_subrun})
    df_subrun.to_csv(
        "log/subrun_combinations_MAGIC" + magic_RunID + "_LST" + lst_RunID + ".txt",
        sep=" ",
        index=None,
    )

subrun_combs = glob.glob("log/subrun_combinations_*txt")
for subrun_comb in subrun_combs:
    df = pd.read_csv(subrun_comb, sep=" ")
    pwd = os.getcwd() + "/"
    magic_RunID = subrun_comb.split("_MAGIC")[1].split("_LST")[0]
    lst_RunID = subrun_comb.split("_LST")[1].split(".txt")[0]

    for i in range(len(df)):
        magic_run = df["magic_subrun"].values[i]
        lst_run = df["lst_subrun"].values[i]
        print("--" + str(i + 1) + "/" + str(len(df)) + "--")
        print("MAGIC: subrun:", magic_run, "&", "LST subrun:", lst_run)
        magic_run, lst_run = str(magic_run).zfill(3), str(lst_run).zfill(4)
        magic_path = (
            magic_dir_name + "/dl1_MAGIC.Run" + magic_RunID + "." + magic_run + "/"
        )
        if os.path.isdir(magic_path) == False:
            os.mkdir(magic_path)
            # Make a MAGIC directory for input it in lst_magic_event_coincidence.py
            os.system(
                "ln -s "
                + pwd
                + file_dl1_dir
                + "/MAGIC/dl1_MAGIC.Run"
                + magic_RunID
                + "."
                + magic_run
                + ".h5 "
                + magic_path
            )

        """
        # Run the coincidence script
        # PLEASE REWRITE THIS according to your analysis environment (e.g., SLURM)
        output = subprocess.run(
            [
                "python",
                "lst1_magic_event_coincidence.py",
                "-l",
                lst_dir_name + "/dl1_LST-1.Run" + lst_RunID + "." + lst_run + ".h5",
                "-m",
                magic_path,
                "-c",
                config,
                "-t",
                outdir,
                "-o",
                file_dl1_dir
                # JS: you can use f"...{...}...." string formating for more compact command
                + "/dl1_coincidence"
                + "/dl1_MAGIC.Run"
                + magic_RunID
                + "."
                + magic_run
                + "_LST-1.Run"
                + lst_RunID
                + "."
                + lst_run
                + ".h5",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Make a log file
        with open(
            "log/coincidence_magic_" + magic_run + "_lst_" + lst_run + ".txt", "w"
        ) as f:
            f.write(output.stderr)

        if os.path.isdir(magic_path) == True:
            os.remove(
                magic_path + "/dl1_MAGIC.Run" + magic_RunID + "." + magic_run + ".h5"
            )
            os.rmdir(magic_path)

        """
        # In the case of slurm enviroment

        subprocess.run(
            [
                "sbatch",
                "job_wrapper.sh",
                lst_dir_name + "/dl1_LST-1.Run" + lst_RunID + "." + lst_run + ".h5",
                magic_path,
                config,
                outdir,
                file_dl1_dir + "/dl1_coincidence" + "/dl1_MAGIC.Run" + magic_RunID + "." + magic_run + "_LST-1.Run" + lst_RunID + "." + lst_run + ".h5",
            ])


        ## job_wrapper.sh is,
        # python lst1_magic_event_coincidence.py -l $1 -m $2 -c $3 -t $4 -o $5
        
