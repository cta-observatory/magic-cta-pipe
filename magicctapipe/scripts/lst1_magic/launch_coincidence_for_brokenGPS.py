import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

magic_dir_name = "data/MAGIC/"
lst_dir_name = "data/LST1/"

magic_t = []
lst_t = []
start_l, stop_l = [],[]
run_l = []
obs_id_l = []
start_m, stop_m = [],[]
run_m = []
obs_id_m = []

magic_dataset = glob.glob(magic_dir_name+"/*h5")
magic_dataset = sorted(magic_dataset)

for magic_data in magic_dataset:
    data_magic = pd.read_hdf(magic_data, key="events/parameters")
    data_magic["trigger_time"] = data_magic["time_sec"] + data_magic["time_nanosec"] * 1e-9
    start_m.append(min(data_magic["trigger_time"]))
    stop_m.append(max(data_magic["trigger_time"]))
    run_m.append(int(data_magic["obs_id"].mean()))

df_magic = pd.DataFrame({"start":start_m,"stop":stop_m,"run":run_m})
df_magic = df_magic.groupby("run").agg({"start": "min", "stop": "max"}).reset_index()

print(df_magic)

lst_dataset = glob.glob(lst_dir_name+"/*h5")
lst_dataset = sorted(lst_dataset)

for lst_data in lst_dataset:
    data_lst = pd.read_hdf(lst_data, key="/dl1/event/telescope/parameters/LST_LSTCam")
    start_l.append(min(data_lst["trigger_time"]))
    stop_l.append(max(data_lst["trigger_time"]))
    run_l.append(int(data_lst["obs_id"].mean()))

df_lst = pd.DataFrame({"start":start_l,"stop":stop_l,"run":run_l})
df_lst = df_lst.groupby("run").agg({"start": "min", "stop": "max"}).reset_index()

print(df_lst)

def print_gti(df1, df2):
    start1_list, stop1_list, start2_list, stop2_list = df1["start"], df1["stop"], df2["start"], df2["stop"]
    i = 0
    subrun_1, subrun_2 = [], []
    for run_ in df1["run"].values:
        df1_ = df1.query("run==@run_")
        start1, stop1 = df1_["start"], df1_["stop"]
        start1, stop1 = float(start1.iloc[0]), float(stop1.iloc[0])
        j = 0
        for start2, stop2 in zip(start2_list, stop2_list):
            obs_time_magic = stop1-start1
            obs_time_lst = stop2-start2
            if [obs_time_magic > obs_time_lst]:
                 s1, e1, s2, e2 = start1,stop1,start2,stop2
            else:
                 s1, e1, s2, e2 = start2,stop2,start1,stop1
            if (s1 < s2) & (s2 < e1) == True:
                 subrun_1.append(df1_["run"].values[0])
                 subrun_2.append(df2["run"].values[j])
            elif (s1 < e2) & (e2 < e1) == True:
                 subrun_1.append(df1_["run"].values[0])
                 subrun_2.append(df2["run"].values[j])
            j = j+1
        i = i+1
    return subrun_1, subrun_2

magic_run, lst_run = print_gti(df_magic, df_lst)                    
df = pd.DataFrame({"magic_run":magic_run,"lst_run":lst_run})

if os.path.isdir("log")==False:
    os.mkdir("log")

print("-- search subrun coincidence --")
for i in range(0, len(df)):
    lst_RunID = str(df["lst_run"][i]).zfill(5)
    magic_RunID = str(df["magic_run"][i]).zfill(8)

    lst_dataset_group = [l for l in lst_dataset if lst_RunID in l]    
    magic_dataset_group = [m for m in magic_dataset if magic_RunID in m]

    start_l_subrun, stop_l_subrun = [],[]
    subrun_l = []
    
    for input_file in lst_dataset_group:
            data_lst = pd.read_hdf(input_file, key="/dl1/event/telescope/parameters/LST_LSTCam")
            start_l_subrun.append(min(data_lst["trigger_time"]))
            stop_l_subrun.append(max(data_lst["trigger_time"]))
            subrun = int(input_file.split(".")[-2])
            subrun_l.append(subrun)
    df_lst_subrun = pd.DataFrame({"start":start_l_subrun,"stop":stop_l_subrun,"run":subrun_l})
    
    start_m_subrun, stop_m_subrun = [],[]
    subrun_m = []
    
    for input_file in magic_dataset_group:
            data_magic = pd.read_hdf(input_file, key="events/parameters")
            data_magic['trigger_time'] = data_magic['time_sec'] + data_magic['time_nanosec'] * 1e-9
            start_m_subrun.append(min(data_magic["trigger_time"]))
            stop_m_subrun.append(max(data_magic["trigger_time"]))
            subrun = int(input_file.split(".")[-2])
            subrun_m.append(subrun)
    df_magic_subrun = pd.DataFrame({"start":start_m_subrun,"stop":stop_m_subrun,"run":subrun_m})
   
    # Find the corresponding subrun for each file
    magic_subrun, lst_subrun = print_gti(df_magic_subrun, df_lst_subrun)
    df_subrun = pd.DataFrame({"magic_subrun":magic_subrun,"lst_subrun":lst_subrun})
    df_subrun.to_csv("log/subrun_combinations_MAGIC"+magic_RunID+"_LST"+lst_RunID+".txt",sep=" ",index=None)

import subprocess

subrun_combs = glob.glob("log/subrun_combinations_*txt")
for subrun_comb in subrun_combs:
    df = pd.read_csv(subrun_comb,sep=" ")
    pwd = os.getcwd()
    magic_RunID = subrun_comb.split("_MAGIC")[1].split("_LST")[0]
    lst_RunID = subrun_comb.split("_LST")[1].split(".txt")[0]
 
    for i in range(len(df)):
        magic_run = df["magic_subrun"].values[i]
        lst_run = df["lst_subrun"].values[i]
        print("--"+str(i+1)+"/"+str(len(df))+"--")
        print("MAGIC: subrun:",magic_run,"&","LST subrun:",lst_run)
        magic_run, lst_run = str(magic_run).zfill(3), str(lst_run).zfill(4)
        magic_path = magic_dir_name+"/dl1_MAGIC.Run"+magic_RunID+"."+magic_run+"/"
        if os.path.isdir(magic_path)==False:
            os.mkdir(magic_path)
            # Make a MAGIC directory for input it in lst_magic_event_coincidence.py
            os.system("ln -s "+pwd+"/data/MAGIC/dl1_MAGIC.Run"+magic_RunID+"."+magic_run+".h5 "+magic_path)
        
        # Run the coincidence script in your local enviroment
        output = subprocess.run(["python","lst1_magic_event_coincidence.py","-l",lst_dir_name+"/dl1_LST-1.Run"+lst_RunID+"."+lst_run+".h5","-m",magic_path,"-c","./config.yaml","-t","init_time_offset/","-o","data/dl1_coincidence/"+"magic_"+magic_run+"_lst_"+lst_run], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
 
        # Make a log file
        with open("log/coincidence_magic_"+magic_run+"_lst_"+lst_run+".txt","w") as f:
            f.write(output.stderr)
        
        if os.path.isdir(magic_path)==True:
            os.remove(magic_path+"/dl1_MAGIC.Run"+magic_RunID+"."+magic_run+".h5")
            os.rmdir(magic_path)