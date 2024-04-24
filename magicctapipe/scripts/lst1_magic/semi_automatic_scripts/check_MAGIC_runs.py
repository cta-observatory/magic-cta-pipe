#This script allows us to get information about every MAGIC run ID (and subruns) in files used for common data analysis (MAGIC1, MAGIC2, LST1). 

#The MAGIC files that can be used for analysis are located in the IT cluster in the following directory:
#/fefs/onsite/common/MAGIC/data/M{tel_id}/event/Calibrated/{YYYY}/{MM}/{DD}

#In this path, 'tel_id' refers to the telescope ID, which must be either 1 or 2. 'YYYY', 'MM', and 'DD' specify the date.

#In the first step, we have to load a dataframe that contains information about the date, the name of the source, and the range of MAGIC #runs. The file in file_path was generated using the spreadsheet "Common MAGIC LST1 data".

import pandas as pd
from datetime import datetime, timedelta
import os
import re

file_path = '/fefs/aswg/workspace/joanna.wojtowicz/data/magic_first_and_last_runs.csv'
df = pd.read_csv(file_path,sep='\t', dtype={'Date (LST conv.)': str, 'Source': str, 'First run': int, 'Last run': int})

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
    
    #Between 2022/09/04 - 2022/12/14 MAGIC 1 had a failure.
    
    YYYY = new_date[:4]
    MM = new_date[4:6]
    DD = new_date[6:8]
    
    results_count = {}

    path = f"/fefs/onsite/common/MAGIC/data/M{tel_id}/event/Calibrated/{YYYY}/{MM}/{DD}"
    

    if os.path.exists(path):
        files = os.listdir(path)
        for filename in files:
            if source in filename:
                results = check_run_ID(path, filename, first_run, last_run, date, source)
                #We will see many results because a file with a run ID has subruns.
                #We must count the same results to get information how many subruns we have.
                for result in results:
                    if result in results_count:
                        results_count[result] += 1
                    else:
                        results_count[result] = 1
    else:
        print(f"No such file or directory: {date}")
    
    for result, count in results_count.items():
        print(f"M{tel_id} \t {result} \t {count}")

tel_id = [1, 2]

for tel in tel_id:
    print()
    print(f"Telescope ID \t Date (LST convention) \t Source \t Run ID \t Subruns")
    for index, row in df.iterrows():
        check_directory(row['Date (LST conv.)'], row['Source'], row['First run'], row['Last run'], tel)
        
