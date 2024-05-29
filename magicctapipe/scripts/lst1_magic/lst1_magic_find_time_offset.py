import numpy as np
import pandas as pd
import sys
import glob
import os
import subprocess
from magicctapipe.io import find_offset

file_dl1_dir = sys.argv[1]

magic_run = int(pd.read_hdf(glob.glob(file_dl1_dir+"/MAGIC/dl1_MAGIC.*Run*h5")[0], key="events/parameters")["obs_id"].mean())
lst_run = int(pd.read_hdf(glob.glob(file_dl1_dir+"//LST1/dl1_LST-1.Run*h5")[0], key="/dl1/event/telescope/parameters/LST_LSTCam")["obs_id"].mean())
magic_run, lst_run = str(magic_run).zfill(8), str(lst_run).zfill(5)

i = 0
for lst_subrun_file in sorted(glob.glob(file_dl1_dir+"/LST1/dl1_LST*h5")):
    df_lst_subrun_file = pd.read_hdf(lst_subrun_file, key="/dl1/event/telescope/parameters/LST_LSTCam")
    df_lst_subrun_file = df_lst_subrun_file.query("intensity>100")
    df_lst_subrun_file = df_lst_subrun_file[["trigger_time","event_type"]]
    if (i>0):
    	df_lst_subrun_file_2 = pd.concat([df_lst_subrun_file_2, df_lst_subrun_file])
    else:
    	df_lst_subrun_file_2 = df_lst_subrun_file
    i=i+1
#lst_merged_file = file_dl1_dir+"/LST1/"+lst_run+".h5"
#df_lst_subrun_file_2.to_hdf(lst_merged_file, key="/dl1/event/telescope/parameters/LST_LSTCam")

outdir="init_time_offset"
try:
    os.mkdir(outdir)
except FileExistsError:
    pass

for magic_subrun_file in sorted(glob.glob(file_dl1_dir+"/MAGIC/dl1_MAGIC.Run"+magic_run+".*.h5")):
    data_magic = pd.read_hdf(magic_subrun_file, key="events/parameters")
    data_magic = data_magic.query("intensity>100")
    data_magic['trigger_time'] = data_magic['time_sec'] + data_magic['time_nanosec'] * 1e-9
    data_magic_m1 = data_magic.query("tel_id==2")
    data_magic_m1.reset_index(inplace=True)
    data_magic_m2 = data_magic.query("tel_id==3")
    data_magic_m2.reset_index(inplace=True)
    magic_subrun_base = magic_subrun_file.split("/")[-1].split(".h5")[0]
    
    data_lst = df_lst_subrun_file_2#pd.read_hdf(lst_merged_file, key="/dl1/event/telescope/parameters/LST_LSTCam")
    
    for M1_M2 in ["M1","M2"]:
        if M1_M2 == "M1":
            data_magic = data_magic_m1
        else:
            data_magic = data_magic_m2

        min_t_magic, max_t_magic = min(data_magic["trigger_time"]), max(data_magic["trigger_time"])
        min_t_lst, max_t_lst = min(data_lst["trigger_time"]), max(data_lst["trigger_time"])

        if min_t_magic < min_t_lst:
            if max_t_magic < max_t_lst: 
                data_magic = data_magic.query("@min_t_lst < trigger_time < @max_t_magic")
            else: 
                data_magic = data_magic.query("@min_t_lst < trigger_time < @max_t_lst")

        if len(data_magic)>0:
            N_begin = data_magic.index[0]
            N_final = data_magic.index[-1]
       
            N_start_ = N_begin#*n_bin
            N_end_  = N_start_+15
            outfile = outdir+"/"+magic_subrun_base+"_"+str(N_start_)+"_init.npy"
            t_magic_all, N_start_out, time_offset_best, n_coincident = find_offset(data_magic,data_lst,N_start=N_start_,N_end=N_end_)
            
            if n_coincident!=0:
                np.save(outfile.replace("MAGIC",M1_M2), np.array([np.mean(t_magic_all), time_offset_best, time_offset_best, n_coincident]))
                time_offset_center = np.load(outfile.replace("MAGIC",M1_M2))[2]
                N_end_of_run = N_final #N_end_+(n_bin-15)
                print("Simultaneous MAGIC+LST1 obs from MAGIC evt index",N_start_,"to",N_end_of_run)
                outfile2 = outdir+"/"+magic_subrun_base+"_"+str(N_start_)+"_"+str(N_end_of_run)+"_detail.npy"
                t_magic_all, time_offset_best, time_offset_best, n_coincident = find_offset(data_magic,data_lst,N_start=N_start_,N_end=N_end_of_run,initial_time_offset=time_offset_center)
                np.save(outfile2.replace("MAGIC",M1_M2), np.array([t_magic_all, time_offset_best,time_offset_best,n_coincident]))
