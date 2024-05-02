#This script allows us to get information about every MAGIC run ID (and subruns) in files used for common data analysis (MAGIC1, MAGIC2, LST1). 

#The MAGIC files that can be used for analysis are located in the IT cluster in the following directory:
#/fefs/onsite/common/MAGIC/data/M{tel_id}/event/Calibrated/{YYYY}/{MM}/{DD}

#In this path, 'tel_id' refers to the telescope ID, which must be either 1 or 2. 'YYYY', 'MM', and 'DD' specify the date.

#In the first step, we have to load a dataframe that contains information about the date, the name of the source, and the range of MAGIC #runs. The file in file_path was generated using the spreadsheet "Common MAGIC LST1 data".

import pandas as pd
from datetime import datetime, timedelta
import os
import re

def table_first_last_run(df):
    df_selected_data = df.iloc[:, [2, 1, 5, 6]]
    df_selected_data.columns = ['DATE','source', 'MAGIC_first_run', 'MAGIC_last_run']
    grouped_data = df_selected_data.groupby(['DATE', 'source'])
    
    result_table = []

    for (date, source), group in grouped_data:
        First_run = group['MAGIC_first_run'].min()
        Last_run = group['MAGIC_last_run'].max()
    
        result_table.append({
            'Date (LST conv.)': date,
            'Source': source,
            'First run': First_run,
            'Last run': Last_run
        })
    
    result = pd.DataFrame(result_table)
    
    return(result)

def check_run_ID(path, filename, first_run, last_run, date, source):
    Y = f'_Y_{source}' 
    #'Y' because we have to be sure that the function counts right filename.

    if Y in filename:
        # Extract run_ids from filename and check range
        run_ids = [int(filename.split("_")[2].split(".")[0])]
        magic_runs = []
    
        for id in run_ids:
            if first_run <= id <= last_run:
                matched = True
                magic_runs.append(f"{date} \t {source} \t {id}")
                #print(f"{date} \t {source} \t {id}")
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
        count_with_source = 0 
        for filename in files:
            if source in filename:
                count_with_source += 1
                results = check_run_ID(path, filename, first_run, last_run, date, source)
                #We will see many results because a file with a run ID has subruns.
                #We must count the same results to get information how many subruns we have.
                for result in results:
                    if result in results_count:
                        results_count[result] += 1
                    else:
                        results_count[result] = 1
        if count_with_source == 0:  
            if(tel_id == 1):
                #Between 2022/09/04 - 2022/12/14 MAGIC 1 had a failure. Therefore we have to skip the range when we want to get information about missing files.
                if(date<'20220904' or date>'20221214'):
                    print(f"No files found containing the source '{source}' on {date}, (M{tel_id})")
            if(tel_id == 2):
                print(f"No files found containing the source '{source}' on {date}, (M{tel_id})")
                        
    else:
        print(f"No such file or directory: {date}")
    
    for result, count in results_count.items():
        print(f"M{tel_id} \t {result} \t {count}")

df = pd.read_hdf( '/fefs/aswg/workspace/federico.dipierro/simultaneous_obs_summary.h5', key='str/table')
database = table_first_last_run(df)
tel_id = [1, 2]

for tel in tel_id:
    print()
    print(f"Telescope ID \t Date (LST convention) \t Source \t Run ID \t Subruns")
    for index, row in database.iterrows():
        check_directory(row['Date (LST conv.)'], row['Source'], row['First run'], row['Last run'], tel)
        
