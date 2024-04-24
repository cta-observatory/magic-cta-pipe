#This script allows us to get information about every MAGIC run ID (and subruns) in files used for common data analysis (MAGIC1, MAGIC2, #LST1). You can also run the script using Jupyter Notebook.

#The MAGIC files that can be used for analysis are located here:
#/fefs/onsite/common/MAGIC/data/M{tel_id}/event/Calibrated/{year}/{month}/{day}

#In this path, 'tel_id' refers to the telescope ID, which must be either 1 or 2. 'Year,' 'month,' and 'day' specify the date.

#In the first step, we have to load a dataframe that contains information about the date, the name of the source, and the range of MAGIC #runs. The file in file_path was generated using the spreadsheet (Common MAGIC LST1 data) from the following link:

#https://docs.google.com/spreadsheets/d/1Tya0tlK-3fuN6_vXOU5FwJruis5vBA9FALAPLFHOfBQ/edit#gid=1066216668

import pandas as pd
from datetime import datetime, timedelta
import os
import re

file_path = '/fefs/aswg/workspace/joanna.wojtowicz/data/magic_first_and_last_runs.csv'
df = pd.read_csv(file_path,sep='\t', dtype={'Date (LST conv.)': str, 'Source': str, 'First run': int, 'Last run': int})

#df

def check_run_ID(path, filename, first_run, last_run, date, source):
    # Extract numbers from filename and check range
    run_ids = [int(s) for s in re.findall(r'\d+', filename)]
    matched = False
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
    
    #Between 2022/09/04 - 2022/12/14 MAGIC 1 had a failure. Therefore we have to skip the range when we want to get information about missing files.
    M1_start_failure = datetime.strptime('20220904', '%Y%m%d')
    M1_end_failure = datetime.strptime('20221214', '%Y%m%d')
    
    year = new_date[:4]
    month = new_date[4:6]
    day = new_date[6:8]
    
    results_count = {}

    path = f"/fefs/onsite/common/MAGIC/data/M{tel_id}/event/Calibrated/{year}/{month}/{day}"
    

    if os.path.exists(path):
        files = os.listdir(path)
        for filename in files:
            if source in filename:
                results = check_run_ID(path, filename, first_run, last_run, date, source)
                #We will see many results becuse a file with a run ID has subruns.
                #We must count the same results to get information how many subruns we have.
                for result in results:
                    if result in results_count:
                        results_count[result] += 1
                    else:
                        results_count[result] = 1
    #else:
        #print(f"No such file or directory: {date}")
    
    for result, count in results_count.items():
        print(f"{result} \t {count}")
    
print(f'For the MAGIC 1 telescope:')
print(f"Date (LST convention) \t Source \t Run ID \t Subruns")

for index, row in df.iterrows():
    check_directory(row['Date (LST conv.)'], row['Source'], row['First run'], row['Last run'], tel_id=1)

print()
print()
print(f'For the MAGIC 2 telescope:')
print(f"Date (LST convention) \t Source \t Run ID \t Subruns")

for index, row in df.iterrows():
    check_directory(row['Date (LST conv.)'], row['Source'], row['First run'], row['Last run'], tel_id=2)


