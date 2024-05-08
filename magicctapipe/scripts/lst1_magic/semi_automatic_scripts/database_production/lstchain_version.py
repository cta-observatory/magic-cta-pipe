"""
Fill the lstchain_0.9 and lstchain_0.10 columns of the LST database (i.e., which version of data is on the IT cluster)
"""


import os

import pandas as pd
import numpy as np
import glob
from string import ascii_letters

def main():

    """
    Main function
    """

    df_LST = pd.read_hdf(
        "/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5",
        key="joint_obs",
    )
    
    
    for i, row in df_LST.iterrows():

        version=[]
        run = row["LST1_run"]
        run = format(int(run), "05d")
        date = row["DATE"]
        directories_version=[i.split('/')[-1] for i in glob.glob(f"/fefs/aswg/data/real/DL1/{date}/v*")]
        
        v_number=np.sort([float(i.replace('v0.','').rstrip(ascii_letters).split('_')[0]) for i in directories_version]).tolist()
        
        
        
        v_number=[str(i).replace('.0','') for i in v_number]
        
        for vers in v_number:
            
            if os.path.isfile(
                f"/fefs/aswg/data/real/DL1/{date}/v0.{vers}/tailcut84/dl1_LST-1.Run{run}.h5"
            ):
                if vers not in version:
                    version.append(vers)
        
        version=list(version)
        
       
        version=[f'v0.{i}'for i in version]
        
        if len(version)>0:
            
            df_LST.loc[i,'last_lstchain_file']=f"/fefs/aswg/data/real/DL1/{date}/{version[-1]}/tailcut84/dl1_LST-1.Run{run}.h5"
        else:
            df_LST.loc[i,'last_lstchain_file']=f"/fefs/aswg/data/real/DL1/{date}/{version}/tailcut84/dl1_LST-1.Run{run}.h5"
        
        df_LST.loc[i, "lstchain_versions"] = str(version)
   
    df_LST.to_hdf(
        "/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5",
        key="joint_obs",
        mode="w",
        min_itemsize={'lstchain_versions':20, 'last_lstchain_file':90,'processed_lstchain_file':90}
    )


if __name__ == "__main__":
    main()
