import os
from magicctapipe.io.io import get_dl2_mean, get_stereo_events
import pytest
import numpy as np
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.table import Table
import pandas as pd
from pathlib import Path
import pexpect




@pytest.fixture(scope="session")
def base_path():
    #TO BE CHANGED
    return Path('/home/evisentin/.cache/magicctapipe')

@pytest.fixture(scope="session")
def stereo_path(base_path):
    p= base_path / 'stereo'
    return p
@pytest.fixture(scope="session")
def dl2_path(base_path):
    p= base_path / 'dl2'
    return p


@pytest.fixture(scope="session")
def dl2_test(dl2_path):
    """
    Toy DL2
    """
    path = dl2_path / "dl2_test.h5"
    data=Table()
    data['obs_id']=[1,1,2,2,2,3,3]
    data['event_id']=[7,7,8,8,8,7,7]
    data['tel_id']=[1,2,1,2,3,1,3]
    data['combo_type']=[1,1,3,3,3,2,2]
    data['multiplicity']=[2,2,3,3,3,2,2]
    data['timestamp']=[1,1,4,4,4,10,10]
    data["pointing_alt"]=[0.6,0.6,0.7,0.7,0.7,0.5,0.5]
    data["pointing_az"]=[1.5,1.5,1.5,1.5,1.5,1.5,1.5]
    data["reco_energy"]=[1,1,1,1,1,1,1]
    data["gammaness"]=[0.5,0.5,1,0.3,0.5,1,1]
    data["reco_alt"]=[40,40,41,41,41,42,42]
    data["reco_az"]=[85,85,85,85,85,85,85]
    write_table_hdf5(data, path, "/events/parameters", 
                     overwrite=False, serialize_meta=False)
    
    return path

@pytest.fixture(scope='session')
def stereo_url():
    #BETTER SOLUTION?
    p=Path('cp02:/fefs/aswg/workspace/elisa.visentin/git_test')
    q=p / 'dl1_stereo_gamma_zd_37.661deg_az_270.641deg_LST-1_MAGIC_run101.h5'
    return q
""" @pytest.fixture(scope='session')
def dl2_url():
    #BETTER SOLUTION?
    p=Path('cp02:/fefs/aswg/workspace/elisa.visentin/git_test')
    q=p / ''
    return q
 """
@pytest.fixture(scope='session')
def env_prefix():
    #ENVIRONMENT VARIABLES TO BE CREATED
    return "MAGIC_CTA_DATA_"


def scp_file(stereo_path,stereo_url, env_prefix):
    """
    Download of test files through scp
    """
    pwd=os.environ[env_prefix + "PASSWORD"]
    pwd.replace('\'','')

    if not (stereo_path / stereo_url.name).exists():

        print("DOWNLOADING...")
        cmd = f'''/bin/bash -c "rsync {str(stereo_url)} {str(stereo_path)} "'''
        if 'rm ' in cmd:
            print ("files cannot be removed")
            exit()
        usr=os.environ[env_prefix + "USER"]
        childP = pexpect.spawn(cmd,timeout=100)
        childP.sendline(cmd)
        childP.expect([f"{usr}@10.200.100.2\'s password:"])
        childP.sendline(pwd)      
        childP.expect(pexpect.EOF)
        childP.close()
    else:
        print("FILE ALREADY EXISTS")
@pytest.fixture(scope='session')
def stereo_file(stereo_path,stereo_url, env_prefix):
        
    scp_file(stereo_path,stereo_url, env_prefix)
    
    name1=stereo_url.name
    url1 = stereo_path / name1
    
    return url1

""" @pytest.fixture(scope='session')
def dl2_file_mc(dl2_path,dl2_url, env_prefix):
    scp_file(dl2_path,dl2_url, env_prefix)
    
    name1=dl2_url.name
    url1 = dl2_path / name1
    
    return url1

    
@pytest.fixture(scope='session')
def dl2_file_real(dl2_path,dl2_url, env_prefix):
    scp_file(dl2_path,dl2_url, env_prefix)
    
    name1=dl2_url.name
    url1 = dl2_path / name1
    
    return url1

"""



def test_get_stereo_events(stereo_file):
    """
    Check on stereo data reading
    """
    event_data = pd.read_hdf(str(stereo_file), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    data=get_stereo_events(event_data)
    assert np.all(data['multiplicity']>1)
    assert np.all(data['combo_type']>=0)

def test_get_stereo_events_cut(stereo_file):
    """
    Check on quality cuts
    """
    event_data = pd.read_hdf(str(stereo_file), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    data=get_stereo_events(event_data, 'intensity>50')
    assert np.all(data['intensity']>50)
"""   
def test_get_dl2_mean_mc(dl2_file_mc):
    event_data = pd.read_hdf(str(stereo_file), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    events=get_dl2_mean(event_data)
    assert "true_energy" in events.colnames
    assert events["multiplicity"].dtype == np.int

def test_get_dl2_mean_real(dl2_file_real):
    event_data = pd.read_hdf(str(stereo_file), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    events=get_dl2_mean(event_data)
    assert "timestamp" in events.colnames

def test_get_dl2_mean_avg(dl2_test):
    event_data = pd.read_hdf(str(stereo_file), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    events=get_dl2_mean(event_data)
    assert events['gammaness']==[0.5,0.6,1]

def test_get_dl2_mean_exc(dl2_file_mc):  
    weight="abc"  
    event_data = pd.read_hdf(str(stereo_file), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    with pytest.raises(ValueError, match=f"Unknown weight type '{weight}'."):
        events=get_dl2_mean(event_data,weight_type=weight) """