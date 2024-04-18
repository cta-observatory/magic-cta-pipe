import pandas as pd
import numpy as np
import os

def main():

    """
    Main function
    """

    
    df = pd.read_hdf(
        "/fefs/aswg/workspace/federico.dipierro/simultaneous_obs_summary.h5", key="/str"
    )  # TODO: put this file in a shared folder

    needed_cols=['source', 'DATE', 'LST1_run', 'MAGIC_stereo', 'MAGIC_trigger', 'MAGIC_HV']
    df_cut=df[needed_cols]
    print(df_cut.columns)
    df_cut['nsb']=np.repeat(np.nan,len(df_cut))
    df_cut['error_code']=np.repeat(np.nan,len(df_cut))
    print(df_cut)
    if os.path.isfile('/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5'):
        df_old=pd.read_hdf('/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5', key="joint_obs")
        df_cut=pd.concat([df_old, df_cut]).drop_duplicates(keep='first')
        df_cut= df_cut.sort_values(by=["DATE","source"])
        print(df_cut)                 # TODO check if fine with update and nsb
    
    df_cut.to_hdf("/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5", key="joint_obs", mode="w")

if __name__ == "__main__":
    main()
