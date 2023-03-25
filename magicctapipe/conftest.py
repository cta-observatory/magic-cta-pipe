import os
import pytest
import numpy as np
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.table import Table
import pandas as pd
from pathlib import Path
import pexpect


"""
Local paths
"""


@pytest.fixture(scope="session")
def base_path():
    # TO BE CHANGED
    return Path("/home/elisa/.cache/magicctapipe")


@pytest.fixture(scope="session")
def stereo_path(base_path):
    p = base_path / "stereo"
    return p


@pytest.fixture(scope="session")
def stereo_path_exc(base_path):
    p = base_path / "stereo_exc"
    return p


@pytest.fixture(scope="session")
def dl2_path(base_path):
    p = base_path / "dl2"
    return p


@pytest.fixture(scope="session")
def dl1_lst_path(base_path):
    p = base_path / "dl1_lst"
    return p


@pytest.fixture(scope="session")
def dl1_magic_path(base_path):
    p = base_path / "dl1_magic"
    return p


@pytest.fixture(scope="session")
def dl1_magic_exc_path(base_path):
    p = base_path / "dl1_magic_exc"
    return p


@pytest.fixture(scope="session")
def irf_path(base_path):
    p = base_path / "irf"
    return p


@pytest.fixture(scope="session")
def irf_path_exc(base_path):
    p = base_path / "irf_exc"
    return p


@pytest.fixture(scope="session")
def pd_path(base_path):
    p = base_path / "pandas"
    return p


"""
Custom data
"""


@pytest.fixture(scope="session")
def dl2_test(dl2_path):
    """
    Toy DL2
    """
    path = dl2_path / "dl2_test.h5"
    data = Table()
    data["obs_id"] = [1, 1, 2, 2, 2, 3, 3]
    data["event_id"] = [7, 7, 8, 8, 8, 7, 7]
    data["tel_id"] = [1, 2, 1, 2, 3, 1, 3]
    data["combo_type"] = [1, 1, 3, 3, 3, 2, 2]
    data["multiplicity"] = [2, 2, 3, 3, 3, 2, 2]
    data["timestamp"] = [1, 1, 4, 4, 4, 10, 10]
    data["pointing_alt"] = [0.6, 0.6, 0.7, 0.7, 0.7, 0.5, 0.5]
    data["pointing_az"] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    data["reco_energy"] = [1, 1, 1, 1, 1, 1, 1]
    data["gammaness"] = [0.5, 0.5, 1, 0.3, 0.5, 1, 1]
    data["reco_alt"] = [40, 40, 41, 41, 41, 42, 42]
    data["reco_az"] = [85, 85, 85, 85, 85, 85, 85]
    write_table_hdf5(
        data, str(path), "/events/parameters", overwrite=True, serialize_meta=False
    )
    return path


@pytest.fixture(scope="session")
def pd_test(pd_path):
    """
    Toy dataframe
    """
    df = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    return df


"""
Remote paths (to download test files)
"""


@pytest.fixture(scope="session")
def base_url():
    # BETTER SOLUTION? CHANGED?
    return Path("cp02:/fefs/aswg/workspace/elisa.visentin/git_test")


@pytest.fixture(scope="session")
def stereo_url(base_url):
    q = base_url / "dl1_stereo_gamma_zd_37.661deg_az_270.641deg_LST-1_MAGIC_run101.h5"
    return q


@pytest.fixture(scope="session")
def dl2_mc_url(base_url):
    q = base_url / "dl2_gamma_zd_35.904deg_az_17.46deg_LST-1_MAGIC_run9502.h5"
    return q


@pytest.fixture(scope="session")
def dl2_real_url(base_url):
    q = base_url / "dl2_LST-1_MAGIC.Run04125.0035.h5"
    return q


@pytest.fixture(scope="session")
def dl1_lst_url(base_url):
    q = base_url / "dl1_LST-1.Run02927.0118.h5"
    return q


@pytest.fixture(scope="session")
def dl1_magic_url(base_url):
    q = base_url / "dl1_M1.Run05093174.001.h5"
    return q


@pytest.fixture(scope="session")
def irf_url(base_url):
    q = (
        base_url
        / "irf_zd_37.814deg_az_270.0deg_software_gh_dyn0.9_theta_glob0.2deg.fits.gz"
    )
    return q


"""
Downloads: useful functions
"""


@pytest.fixture(scope="session")
def env_prefix():
    # ENVIRONMENT VARIABLES TO BE CREATED
    return "MAGIC_CTA_DATA_"


def scp_file(path, url, env_prefix):
    """
    Download of test files through scp
    """
    pwd = os.environ[env_prefix + "PASSWORD"]
    if not (path / url.name).exists():
        print("DOWNLOADING...")
        cmd = f'''/bin/bash -c "rsync {str(url)} {str(path)} "'''
        if "rm " in cmd:
            print("files cannot be removed")
            exit()
        usr = os.environ[env_prefix + "USER"]
        childP = pexpect.spawn(cmd, timeout=500)
        childP.sendline(cmd)
        childP.expect([f"{usr}@10.200.100.2's password:"])
        childP.sendline(pwd)
        childP.expect(pexpect.EOF)
        childP.close()
    else:
        print("FILE ALREADY EXISTS")


"""
Downloads: files
"""


@pytest.fixture(scope="session")
def stereo_file(stereo_path, stereo_url, env_prefix):
    scp_file(stereo_path, stereo_url, env_prefix)
    name1 = stereo_url.name
    url1 = stereo_path / name1
    return url1


@pytest.fixture(scope="session")
def dl2_file_mc(dl2_path, dl2_mc_url, env_prefix):
    scp_file(dl2_path, dl2_mc_url, env_prefix)
    name1 = dl2_mc_url.name
    url1 = dl2_path / name1
    return url1


@pytest.fixture(scope="session")
def dl2_file_real(dl2_path, dl2_real_url, env_prefix):
    scp_file(dl2_path, dl2_real_url, env_prefix)
    name1 = dl2_real_url.name
    url1 = dl2_path / name1
    return url1


@pytest.fixture(scope="session")
def dl1_file_lst(dl1_lst_path, dl1_lst_url, env_prefix):
    scp_file(dl1_lst_path, dl1_lst_url, env_prefix)
    name1 = dl1_lst_url.name
    url1 = dl1_lst_path / name1
    return url1


def dl1_file_magic(dl1_magic_path, dl1_magic_url, env_prefix):
    scp_file(dl1_magic_path, dl1_magic_url, env_prefix)
    name1 = dl1_magic_url.name
    url1 = dl1_magic_path / name1
    return url1


def irf_file(irf_path, irf_url, env_prefix):
    scp_file(irf_path, irf_url, env_prefix)
    name1 = irf_url.name
    url1 = irf_path / name1
    return url1
