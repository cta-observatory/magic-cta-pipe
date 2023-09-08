import pandas as pd
import numpy as np
import os
import yaml

def list_run(source_out, df, skip_LST, skip_MAGIC):
    file_list=[f'{source_out}_LST_runs.txt',f'{source_out}_MAGIC_runs.txt']  #### LST, MAGIC!!!!
    for j in file_list:
        if os.path.isfile(j):
            os.remove(j)
            print(f"{j} deleted.")
    print(skip_LST)
    print(type(skip_LST))
    for k in range(len(df)):
   
    
    
        LST=np.fromstring(df['LST_runs'][k][1:-1],dtype='int', sep=', ')
    
    
        for j in range(len(LST)):
            skip=False
            for jkz in range(len(skip_LST)):
                if int(LST[j])==skip_LST[jkz]: 
                    skip=True 
            if skip==False:
                with open(file_list[0], 'a+') as f:
                    f.write(str(df['date_LST'][k])+','+str(LST[j])+'\n')
        MAGIC_min=int(df['first_MAGIC'][k])   
        MAGIC_max=int(df['last_MAGIC'][k]) 
        for z in range(MAGIC_min,MAGIC_max+1):
            skip=False
            for msp in range(len(skip_MAGIC)):
                if int(z)==skip_MAGIC[msp]:
                    skip=True
            if skip==False:        
                with open(file_list[1], 'a+') as f:
                    f.write(str(df['date_MAGIC'][k])+','+str(z)+'\n')
        
def main():
    with open('config_google.yaml', "rb") as f:   # "rb" mode opens the file in binary format for reading
                config = yaml.safe_load(f)
    source_in=config['data_selection_and_lists']['source_name_database']

    source_out=config['data_selection_and_lists']['source_name_output']
    range=config['data_selection_and_lists']['time_range']
    skip_LST=config['data_selection_and_lists']['skipped_LST_runs']
    skip_MAGIC=config['data_selection_and_lists']['skipped_MAGIC_runs']
    print(skip_LST,'skip')
    print(source_in)
    df=pd.read_hdf('observations.h5',key='joint_obs')
    df=df.astype({'YY_LST':int,'MM_LST':int,'DD_LST':int})




    df['trigger']=df['trigger'].str.rstrip("']")
    df['trigger']=df['trigger'].str.lstrip("['")
    df['HV']=df['HV'].str.rstrip("']")
    df['HV']=df['HV'].str.lstrip("['")
    df['stereo']=df['stereo'].str.rstrip("]")
    df['stereo']=df['stereo'].str.lstrip("[")
    df.query(f'source=="{source_in}" & trigger=="L3T" & HV=="Nominal" & stereo == "True"', inplace=True)
    if range==True:
        Y_min=int(config['data_selection_and_lists']['YY_min'])
        M_min=int(config['data_selection_and_lists']['MM_min'])
        D_min=int(config['data_selection_and_lists']['DD_min'])
        Y_max=int(config['data_selection_and_lists']['YY_max'])
        M_max=int(config['data_selection_and_lists']['MM_max'])
        D_max=int(config['data_selection_and_lists']['DD_max'])
        df.query(f'YY_LST>={Y_min} & MM_LST>={M_min} & DD_LST>={D_min} & YY_LST<={Y_max} & MM_LST<={M_max} & DD_LST<={D_max}', inplace=True)
    if range==False:
        dates=config['data_selection_and_lists']['date_list']
        print(dates)
        df=df[df['date_LST'].isin(dates)]


    df=df.reset_index()
    df=df.drop('index',axis=1)
    df.to_hdf('observations_query.h5',key='joint_obs', mode='w')
    list_run(source_out, df,skip_LST,skip_MAGIC)

if __name__ == "__main__":
    main()
