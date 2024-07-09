#This script allows us to get information about every MAGIC run ID (and subruns) in files (in a time interval) used for common data analysis (MAGIC1, MAGIC2, LST1). 

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

def table_magic_runs(df, date_min, date_max):
    df_selected_data = df.iloc[:, [2, 1, 25]]
    df_selected_data.columns = ['DATE','source', 'MAGIC_runs']
    grouped_data = df_selected_data.groupby(['DATE', 'source'])
    result_table = []

    for (date, source), group in grouped_data:
        if (date>=date_min and date<=date_max):  
            runs_combined = group['MAGIC_runs'].sum()
    
            result_table.append({
                'DATE': date,
                'source': source,
                'MAGIC runs': runs_combined
            })
        
    result = pd.DataFrame(result_table)
    result['MAGIC runs'] = result['MAGIC runs'].apply(fix_lists_and_convert)
    return(result)

def existing_files( tel_id, date, source, magic_runs ):

    magic_runs = str(magic_runs)  
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
            # Counter for files that include the source.  
            # Counter for files that include the run_id.
        for filename in files:
            if date and source and Y in filename:
                count_with_source += 1
                if magic_runs in filename:
                    count_with_run_id += 1
        if count_with_source != 0 and count_with_run_id != 0:
            print(f"{date}\t{source}\t{magic_runs}\t{count_with_run_id}")
                    
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
            if date and source and Y in filename:
                count_with_source += 1
                for runs in magic_runs:
                   # run = str(runs)
                    if run in filename:
                        count_with_run_id += 1
        if count_with_source == 0:  
            if(tel_id == 1):
                #Between 2022/09/04 - 2022/12/14 MAGIC 1 had a failure. Therefore we have to skip the range when we want to get information about missing files.
                if(date<='20220904' or date>='20221214'):
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
    
    #TO DO : set time interval- format YYYYMMDD
    date_min = '20240601'
    date_max = '20240630'
    
    df = pd.read_hdf( '/fefs/aswg/workspace/federico.dipierro/MAGIC_LST1_simultaneous_runs_info/simultaneous_obs_summary.h5', key='str/table')

    tel_id = [1, 2]
    database = table_magic_runs(df, date_min, date_max)
    database_exploded =  database.explode('MAGIC runs')
    database_exploded_reset = database_exploded.reset_index(drop=True)

    for tel in tel_id:
        print(f"MAGIC {tel}")
        print(f"DATE\tsource\tRun ID\t Subruns")
        for index, row in database_exploded_reset.iterrows():
            existing_files(tel, row['DATE'], row['source'], row['MAGIC runs'])
        print()
        for index, row in database.iterrows():
            missing_files(tel, row['DATE'], row['source'], row['MAGIC runs'])
        print()
        
if __name__ == "__main__":
    main()
