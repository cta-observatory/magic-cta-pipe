import argparse
from lstchain.image.modifier import calculate_noise_parameters
import numpy as np
import yaml
import glob
def main():

    """
    create list of LST runs with nsb
    """   
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config_general.yaml",
        help="Path to a configuration file",
    )

    args = parser.parse_args()
    with open(args.config_file, "rb") as f:   # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
   
    simtel='/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/sim_telarray/node_theta_14.984_az_355.158_/output_v1.4/simtel_corsika_theta_14.984_az_355.158_run10.simtel.gz'
    runs = config["general"]["LST_runs"] 
    nsb = config["general"]["nsb"]  
    width=[a/2 - b/2 for a, b in zip(nsb[1:], nsb[:-1])]
    source= config["directories"]['target_name']    
    width.append(0.25)
    nsb_limit=[a + b for a, b in zip(nsb[:], width[:])]
    nsb_limit.insert(0,0)      
    print(nsb_limit)
    lst_config='lstchain_standard_config.json'
    LST_files=np.sort(glob.glob(f'{source}_LST_[0-9]*.txt'))
    with open(runs) as LSTfile:
      LST_runs = np.genfromtxt(LSTfile,dtype=str,delimiter=',')
      if (len(LST_runs)==2) and (len(LST_runs[0])==10):

        LST=LST_runs
        
        LST_runs=[]
        LST_runs.append(LST)
        
      print(LST_runs)

      for i in LST_runs:
        print(i)
        duplicate=False
        for k in LST_files:
            with open(k) as myfl:
                if f'{i[0]},{i[1]}' in myfl.read():
                      print('run already processed')
                      duplicate=True
                      continue
        if duplicate==True:
            continue
        lstObsDir = i[0].split("_")[0]+i[0].split("_")[1]+i[0].split("_")[2]
        inputdir = f'/fefs/aswg/data/real/DL1/{lstObsDir}/v0.9/tailcut84'
        run = np.sort(glob.glob(f"{inputdir}/dl1*Run*{i[1]}.*.h5"))
        noise=[]
        if len(run)==0:
          continue
        if len(run)<25:
            mod=1
        else:
            mod=int(len(run)/25)
        for ii in range (0, len(run)):
          if ii%mod==0:

            a,b,c= calculate_noise_parameters(simtel, run[ii], lst_config)
            noise.append(a)
        a=sum(noise)/len(noise)
        std=np.std(noise)
        print('nsb average (all)',a, 'std', std)
        subrun_ok=[]
        for sr in range (0, len(noise)):
            if np.abs(noise[sr]-a)<3*std:
                subrun_ok.append(noise[sr])
        a=sum(subrun_ok)/len(subrun_ok)
        print('nsb average', a)
        for j in range (0,len(nsb)):
          if (a<nsb_limit[j+1])&(a>nsb_limit[j]):
            with open(f"{source}_LST_{nsb[j]}_.txt", "a+") as f:
                
              f.write(str(i[0])+","+str(i[1])+"\n")  
        
          
if __name__ == "__main__":
    main()          
