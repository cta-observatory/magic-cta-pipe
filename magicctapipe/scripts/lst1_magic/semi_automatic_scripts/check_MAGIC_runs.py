#This script allows us to get information about every MAGIC run ID (and subruns) in files used for common data analysis (MAGIC1, MAGIC2, LST1). 

#The MAGIC files that can be used for analysis are located in the IT cluster in the following directory:
#/fefs/onsite/common/MAGIC/data/M{tel_id}/event/Calibrated/{YYYY}/{MM}/{DD}

#In this path, 'tel_id' refers to the telescope ID, which must be either 1 or 2. 'YYYY', 'MM', and 'DD' specify the date.

#In the first step, we have to load a dataframe that contains information about the date, the name of the source, and the range of MAGIC #runs. The file in file_path was generated using the spreadsheet "Common MAGIC LST1 data".

import pandas as pd
from datetime import datetime, timedelta
import os
import re

def fix_lists_and_convert(cell):
    # Remove brackets to avoid double lists and split on ']['
    parts = cell.replace('][', ',').strip('[]').split(',')
    return list(dict.fromkeys(int(item) for item in parts))

def table_first_last_run(df):
    df_selected_data = df.iloc[:, [2, 1, 5, 6, 25]]
    df_selected_data.columns = ['DATE','source', 'MAGIC_first_run', 'MAGIC_last_run', 'MAGIC_runs']
    grouped_data = df_selected_data.groupby(['DATE', 'source'])
    result_table = []

    for (date, source), group in grouped_data:
        First_run = group['MAGIC_first_run'].min()
        Last_run = group['MAGIC_last_run'].max()
        runs_combined = group['MAGIC_runs'].sum()
    
        result_table.append({
            'DATE': date,
            'source': source,
            'First run': First_run,
            'Last run': Last_run,
            'MAGIC runs': runs_combined
        })
    
    result = pd.DataFrame(result_table)
    result['MAGIC runs'] = result['MAGIC runs'].apply(fix_lists_and_convert)
    return(result)

def check_run_ID(path, filename, first_run, last_run, date, source, tel_id):

    #We have to be sure that the function counts right filename.
    date_obs = filename.split("_")[0]
    run = filename.split("_")[2].split(".")[0]
    subrun = filename.split("_")[2].split(".")[1]
    Y = f'{date_obs}_M{tel_id}_{run}.{subrun}_Y_{source}' 
    r = f".root"

    if Y and r in filename:
        # Extract run_ids from filename and check range
        run_ids = [int(filename.split("_")[2].split(".")[0])]
        magic_runs = []
    
        for id in run_ids:
            if first_run <= id <= last_run:
                magic_runs.append(f"{date}\t{source}\t{id}")
        return magic_runs
        
def check_directory(date, source, first_run, last_run, tel_id):
     # In the table date are written as follows: YYYYMMDD, for example '20191123' We need a datetime object.
    date_obj = datetime.strptime(date, '%Y%m%d')

     # Date in MAGIC convention ( 'LST +1 day')
    date_obj += timedelta(days=1)
    new_date = datetime.strftime(date_obj, '%Y%m%d')
    
    YYYY = new_date[:4]
    MM = new_date[4:6]
    DD = new_date[6:8]
    
    results_count = {}

    path = f"/fefs/onsite/common/MAGIC/data/M{tel_id}/event/Calibrated/{YYYY}/{MM}/{DD}"
    
    if os.path.exists(path):
        files = os.listdir(path)
        
        for filename in files:
            if source in filename:
                results = check_run_ID(path, filename, first_run, last_run, date, source, tel_id)
                #We will see many results becuse a file with a run ID has subruns.
                #We must count the same results to get information how many subruns we have.
                for result in results:
                    if result in results_count:
                        results_count[result] += 1
                    else:
                        results_count[result] = 1
    
    for result, count in results_count.items():
        print(f"{result}\t{count}")

def missing_files( tel_id, date, source, magic_runs ):
    
    for runs in magic_runs:
        run = str(runs)
    
    date_obj = datetime.strptime(date, '%Y%m%d')
    date_obj += timedelta(days=1)
    new_date = datetime.strftime(date_obj, '%Y%m%d')
    YYYY = new_date[:4]
    MM = new_date[4:6]
    DD = new_date[6:8]
    Y = f"_Y_"
    
    path = f"/fefs/onsite/common/MAGIC/data/M{tel_id}/event/Calibrated/{YYYY}/{MM}/{DD}" 
    
    if os.path.exists(path):
        files = os.listdir(path)
        count_with_source = 0  
        count_with_run_id = 0
        # Counter for files that include the source. We want to check if any file with the source was found. 
        # Counter for files that include the run_id. We want to check if any file with the run_id was found. 
        for filename in files:
            if source in filename:
                count_with_source += 1
                for runs in magic_runs:
                   # run = str(runs)
                    if run in filename:
                        count_with_run_id += 1
        if count_with_source == 0:  
            if(tel_id == 1):
                #Between 2022/09/04 - 2022/12/14 MAGIC 1 had a failure. Therefore we have to skip the range when we want to get information about missing files.
                if(date<'20220904' or date>'20221214'):
                    print(f"No files found containing the source '{source}' on {date}")
                else:
                    print(f"M1 failure. No files found containing the source '{source}' on {date}.")
            if(tel_id == 2):
                print(f"No files found containing the source '{source}' on {date}")
        if count_with_source != 0 and count_with_run_id == 0:
            if(date<'20220904' or date>'20221214'):
                print(f"No run id: {run} found in the {source} on {date}.")
    else:
        print(f"No such file or directory: {date}")
        
def main():

    df = pd.read_hdf( '/fefs/aswg/workspace/federico.dipierro/MAGIC_LST1_simultaneous_runs_info/simultaneous_obs_summary.h5', key='str/table')

    tel_id = [1, 2]
    database = table_first_last_run(df)

    for tel in tel_id:
        print(f"MAGIC {tel}")
        print(f"DATE\tsource\tRun ID\t Subruns")
        for index, row in database.iterrows():
            check_directory(row['DATE'], row['source'], row['First run'], row['Last run'], tel)
        print()
        for index, row in database.iterrows():
            missing_files(tel, row['DATE'], row['source'], row['MAGIC runs'])
        print()
        
if __name__ == "__main__":
    main()
    


        
