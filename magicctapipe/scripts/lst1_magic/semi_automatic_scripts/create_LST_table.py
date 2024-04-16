import pandas as pd
import numpy as np

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
    df_cut.to_hdf("/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5", key="joint_obs", mode="w")

if __name__ == "__main__":
    main()
