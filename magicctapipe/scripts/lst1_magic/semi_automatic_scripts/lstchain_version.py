"""
Fill the lstchain_0.9 and lstchain_0.10 columns of the LST database (i.e., which version of data is on the IT cluster)
"""


import pandas as pd
import os


def main():
    

    """
    Main function
    """

    df_LST = pd.read_hdf(
        "/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5",
        key="joint_obs",
    )
    for i, row in df_LST.iterrows():
       
        lst_9 = False
        lst_10=False
        run=row['LST1_run']
        run=format(int(run), '05d')
        date=row['DATE']
        
        if os.path.isfile(f'/fefs/aswg/data/real/DL1/{date}/v0.9/tailcut84/dl1_LST-1.Run{run}.h5'):
            lst_9=True
        if os.path.isfile(f'/fefs/aswg/data/real/DL1/{date}/v0.10/tailcut84/dl1_LST-1.Run{run}.h5'):
            lst_10=True
        if (lst_9==False) and (lst_10==False):
            df_LST.at[i,'error_code'] = '002'
        df_LST.at[i,'lstchain_0.9'] = lst_9
        df_LST.at[i,'lstchain_0.10'] = lst_10
        

    
    df_LST.to_hdf(
        "/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5",
        key="joint_obs",
        mode="w",
    )

if __name__ == "__main__":
    main()

























