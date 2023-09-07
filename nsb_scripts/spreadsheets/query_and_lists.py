import pandas as pd
import numpy as np
import os
import yaml

def list_run(source_out, df):
    file_list=[f'{source_out}_LST_runs.txt',f'{source_out}_MAGIC_runs.txt']  #### LST, MAGIC!!!!
    for j in file_list:
        if os.path.isfile(j):
            os.remove(j)
            print(f"{j} deleted.")
    for k in range(len(df)):
   
    
    
        LST=np.fromstring(df['LST_runs'][k][1:-1],dtype='int', sep=', ')
    
    
        for j in range(len(LST)):
        
            with open(file_list[0], 'a+') as f:
                f.write(str(df['date_LST'][k])+','+str(LST[j])+'\n')
        MAGIC_min=int(df['first_MAGIC'][k])   
        MAGIC_max=int(df['last_MAGIC'][k]) 
        for z in range(MAGIC_min,MAGIC_max+1):
            with open(file_list[1], 'a+') as f:
                f.write(str(df['date_MAGIC'][k])+','+str(z)+'\n')
        
def main():
    with open('config_google.yaml', "rb") as f:   # "rb" mode opens the file in binary format for reading
                config = yaml.safe_load(f)
    source_in=config['data_selection_and_lists']['source_name_database']
    source_out=config['data_selection_and_lists']['source_name_output']
    print(source_in)
    df=pd.read_hdf('observations.h5',key='joint_obs')
    df=df.astype({'YY_LST':int,'MM_LST':int,'DD_LST':int})
    df.query(f'source=="{source_in}"', inplace=True)
    df=df.reset_index()
    df=df.drop('index',axis=1)
    
    list_run(source_out, df)

if __name__ == "__main__":
    main()
