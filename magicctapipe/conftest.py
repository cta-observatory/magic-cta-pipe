import os
import pytest
import numpy as np
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.table import Table
import pandas as pd
from pathlib import Path
import pexpect
import subprocess

ntrain = 1  # number of MC train runs
DL0_gamma_data = [
    "simtel_corsika_theta_16.087_az_108.090_run1.simtel.gz",
    "simtel_corsika_theta_16.087_az_108.090_run2.simtel.gz",
]
DL0_gamma_p = [
    "simtel_corsika_theta_16.087_az_108.090_run1.simtel.gz",
    "simtel_corsika_theta_16.087_az_108.090_run2.simtel.gz",
]
"""
Temporary paths
"""


@pytest.fixture(scope="session")
def temp_DL1_gamma(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_gammas")


@pytest.fixture(scope="session")
def temp_DL1_gamma_train(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_gamma_train")


@pytest.fixture(scope="session")
def temp_DL1_gamma_test(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_gamma_test")


@pytest.fixture(scope="session")
def temp_rf(tmp_path_factory):
    return tmp_path_factory.mktemp("RF")


@pytest.fixture(scope="session")
def temp_DL2_gamma(tmp_path_factory):
    return tmp_path_factory.mktemp("DL2_gammas")


@pytest.fixture(scope="session")
def temp_irf(tmp_path_factory):
    return tmp_path_factory.mktemp("IRF")


@pytest.fixture(scope="session")
def temp_DL1_p(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_protons")


@pytest.fixture(scope="session")
def temp_DL1_p_train(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_proton_train")


@pytest.fixture(scope="session")
def temp_DL1_p_test(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_proton_test")


@pytest.fixture(scope="session")
def temp_DL2_p(tmp_path_factory):
    return tmp_path_factory.mktemp("DL2_protons")


@pytest.fixture(scope="session")
def temp_DL2_test(tmp_path_factory):
    return tmp_path_factory.mktemp("DL2_test")


@pytest.fixture(scope="session")
def temp_train_exc(tmp_path_factory):
    return tmp_path_factory.mktemp("train_exc")


@pytest.fixture(scope="session")
def temp_irf_exc(tmp_path_factory):
    return tmp_path_factory.mktemp("irf_exc")


@pytest.fixture(scope="session")
def temp_pandas(tmp_path_factory):
    return tmp_path_factory.mktemp("pandas")


"""
Local paths
"""


@pytest.fixture(scope="session")
def base_path():
    # TO BE CHANGED
    return Path("/home/elisa/.cache/magicctapipe")


@pytest.fixture(scope="session")
def dl0_gamma_path(base_path):
    return base_path / "DL0gamma"


@pytest.fixture(scope="session")
def dl0_p_path(base_path):
    return base_path / "DL0p"


@pytest.fixture(scope="session")
def conf_path(base_path):
    p = base_path / "config"
    return p


"""
Custom data
"""


@pytest.fixture(scope="session")
def dl2_test(temp_DL2_test):
    """
    Toy DL2
    """
    path = temp_DL2_test / "dl2_test.h5"
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
def pd_test():
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
    return Path("cp02:/fefs/aswg/workspace/elisa.visentin/git_test")


@pytest.fixture(scope="session")
def dl0_gamma_url(base_url):
    return base_url / "DL0gamma"


@pytest.fixture(scope="session")
def dl0_p_url(base_url):
    # BETTER SOLUTION? CHANGED?
    return base_url / "DL0p"


@pytest.fixture(scope="session")
def conf_url(base_url):
    q = base_url / "config.yaml"
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
def dl0_gamma(dl0_gamma_url, dl0_gamma_path, env_prefix):
    gamma_dl0 = []
    for file in DL0_gamma_data:
        scp_file(dl0_gamma_path, dl0_gamma_url / f"{file}", env_prefix)
        gamma_dl0.append(dl0_gamma_path / f"{file}")
    return gamma_dl0


@pytest.fixture(scope="session")
def dl0_p(dl0_p_url, dl0_p_path, env_prefix):
    p_dl0 = []
    for file in DL0_gamma_p:
        scp_file(dl0_p_path, dl0_p_url / f"{file}", env_prefix)
        p_dl0.append(dl0_p_path / f"{file}")

    return p_dl0


@pytest.fixture(scope="session")
def config(conf_path, conf_url, env_prefix):
    scp_file(conf_path, conf_url, env_prefix)
    name1 = conf_url.name
    url1 = conf_path / name1
    return url1


"""
Data processing
"""


@pytest.fixture(scope="session")
def gamma_l1(temp_DL1_gamma, dl0_gamma, config):
    """
    Produce a dl1 file
    """

    for file in dl0_gamma:
        subprocess.run(
            [
                "lst1_magic_mc_dl0_to_dl1",
                f"-i{str(file)}",
                f"-o{str(temp_DL1_gamma)}",
                f"-c{str(config)}",
            ]
        )

    return temp_DL1_gamma.glob("*")


@pytest.fixture(scope="session")
def gamma_stereo(temp_DL1_gamma_train, temp_DL1_gamma_test, gamma_l1, config):
    """
    Produce a dl1 stereo file
    """

    for i, file in enumerate(gamma_l1):
        if i < ntrain:
            out = temp_DL1_gamma_train
        else:
            out = temp_DL1_gamma_test
        subprocess.run(
            [
                "lst1_magic_stereo_reco",
                f"-i{str(file)}",
                f"-o{str(out)}",
                f"-c{str(config)}",
            ]
        )

    return (temp_DL1_gamma_train, temp_DL1_gamma_test)


@pytest.fixture(scope="session")
def p_l1(temp_DL1_p, dl0_p, config):
    """
    Produce a dl1 file
    """
    for file in dl0_p:
        subprocess.run(
            [
                "lst1_magic_mc_dl0_to_dl1",
                f"-i{str(file)}",
                f"-o{str(temp_DL1_p)}",
                f"-c{str(config)}",
            ]
        )
    return temp_DL1_p.glob("*")


@pytest.fixture(scope="session")
def p_stereo(temp_DL1_p_train, temp_DL1_p_test, p_l1, config):
    """
    Produce a dl1 stereo file
    """

    for i, file in enumerate(p_l1):
        if i < ntrain:
            out = temp_DL1_p_train
        else:
            out = temp_DL1_p_test
        subprocess.run(
            [
                "lst1_magic_stereo_reco",
                f"-i{str(file)}",
                f"-o{str(out)}",
                f"-c{str(config)}",
            ]
        )
    return (temp_DL1_p_train, temp_DL1_p_test)


@pytest.fixture(scope="session")
def RF(gamma_stereo, p_stereo, temp_rf, config):
    """
    Produce a dl1 stereo file
    """

    subprocess.run(
        [
            "lst1_magic_train_rfs",
            f"-g{str(gamma_stereo[0])}",
            f"-p{str(p_stereo[0])}",
            f"-o{str(temp_rf)}",
            f"-c{str(config)}",
            "--train-energy",
            "--train-disp",
            "--train-classifier",
            "--use-unsigned",
        ]
    )
    return temp_rf


@pytest.fixture(scope="session")
def gamma_dl2(temp_DL1_gamma_test, RF, temp_DL2_gamma):
    """
    Produce a dl1 stereo file
    """

    for file in temp_DL1_gamma_test.glob("*"):
        subprocess.run(
            [
                "lst1_magic_dl1_stereo_to_dl2",
                f"-d{str(file)}",
                f"-r{str(RF)}",
                f"-o{str(temp_DL2_gamma)}",
            ]
        )
    return temp_DL2_gamma.glob("*")


@pytest.fixture(scope="session")
def IRF(gamma_dl2, config, temp_irf):
    """
    Produce a dl1 stereo file
    """

    for file in gamma_dl2:
        subprocess.run(
            [
                "lst1_magic_create_irf",
                f"-g{str(file)}",
                f"-o{str(temp_irf)}",
                f"-c{str(config)}",
            ]
        )
    return temp_irf.glob("*")


@pytest.fixture(scope="session")
def p_dl2(temp_DL1_p_test, RF, temp_DL2_p):
    """
    Produce a dl1 stereo file
    """

    for file in temp_DL1_p_test.glob("*"):
        subprocess.run(
            [
                "lst1_magic_dl1_stereo_to_dl2",
                f"-d{str(file)}",
                f"-r{str(RF)}",
                f"-o{str(temp_DL2_p)}",
            ]
        )
    return temp_DL2_p.glob("*")
