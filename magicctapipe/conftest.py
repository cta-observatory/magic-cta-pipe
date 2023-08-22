import pytest
import numpy as np
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.table import Table
import pandas as pd
import subprocess
from math import trunc
from ctapipe.utils.download import download_file_cached

maxjoint = 13000
maxmonly = 500

DL0_gamma_data = [
    "simtel_corsika_theta_16.087_az_108.090_run1.simtel.gz",  # smaller
    "simtel_corsika_theta_16.087_az_108.090_run2.simtel.gz",  # smaller
    "simtel_corsika_theta_16.087_az_108.090_run3.simtel.gz",  # smaller
    "simtel_corsika_theta_16.087_az_108.090_run4.simtel.gz",  # smaller
    # "simtel_corsika_theta_16.087_az_108.090_run1_all.simtel.gz", #standard
    # "simtel_corsika_theta_16.087_az_108.090_run2_all.simtel.gz", #standard
    # "simtel_corsika_theta_16.087_az_108.090_run1_cut.simtel.gz", #cut
    # "simtel_corsika_theta_16.087_az_108.090_run2_cut.simtel.gz", #cut
]
DL0_p_data = [
    "simtel_corsika_theta_16.087_az_108.090_run1.simtel.gz",
    "simtel_corsika_theta_16.087_az_108.090_run2.simtel.gz",
]
ntraingamma = trunc(len(DL0_gamma_data) / 2)  # number of MC train runs
ntrainp = trunc(len(DL0_p_data) / 2)  # number of MC train runs
DL0_M1_data = [
    # 20201216_M1_05093711.001_Y_CrabNebula-W0.40+035.root",
    # "20201216_M1_05093711.002_Y_CrabNebula-W0.40+035.root",
    # "20201216_M1_05093711.003_Y_CrabNebula-W0.40+035.root",
    # "20201216_M1_05093711.004_Y_CrabNebula-W0.40+035.root",
    "20201216_M1_05093711.014_Y_CrabNebula-W0.40+035.root",
]
DL0_M2_data = [
    # "20201216_M2_05093711.001_Y_CrabNebula-W0.40+035.root",
    # "20201216_M2_05093711.002_Y_CrabNebula-W0.40+035.root",
    # "20201216_M2_05093711.003_Y_CrabNebula-W0.40+035.root",
    # "20201216_M2_05093711.004_Y_CrabNebula-W0.40+035.root",
    "20201216_M2_05093711.014_Y_CrabNebula-W0.40+035.root",
]
DL1_LST_data = ["dl1_LST-1.Run03265.0094.h5"]
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
def temp_DL1_gamma_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_gammas_monly")


@pytest.fixture(scope="session")
def temp_DL1_gamma_train_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_gamma_train_monly")


@pytest.fixture(scope="session")
def temp_DL1_gamma_test_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_gamma_test_monly")


@pytest.fixture(scope="session")
def temp_rf_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("RF_monly")


@pytest.fixture(scope="session")
def temp_DL2_gamma_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL2_gammas_monly")


@pytest.fixture(scope="session")
def temp_irf_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("IRF_monly")


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
def temp_DL1_p_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_protons_monly")


@pytest.fixture(scope="session")
def temp_DL1_p_train_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_proton_train_monly")


@pytest.fixture(scope="session")
def temp_DL1_p_test_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_proton_test_monly")


@pytest.fixture(scope="session")
def temp_DL2_p_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL2_protons_monly")


@pytest.fixture(scope="session")
def temp_DL2_test_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL2_test_monly")


@pytest.fixture(scope="session")
def temp_train_exc(tmp_path_factory):
    return tmp_path_factory.mktemp("train_exc")


@pytest.fixture(scope="session")
def temp_irf_exc(tmp_path_factory):
    return tmp_path_factory.mktemp("irf_exc")


@pytest.fixture(scope="session")
def temp_DL1_M(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_MAGIC")


@pytest.fixture(scope="session")
def temp_DL1_M_merge(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_MAGIC_merge")


@pytest.fixture(scope="session")
def temp_DL1_M_exc(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_MAGIC_exc")


@pytest.fixture(scope="session")
def temp_coinc(tmp_path_factory):
    return tmp_path_factory.mktemp("coincidence")


@pytest.fixture(scope="session")
def temp_coinc_stereo(tmp_path_factory):
    return tmp_path_factory.mktemp("coincidence_stereo")


@pytest.fixture(scope="session")
def temp_DL2_real(tmp_path_factory):
    return tmp_path_factory.mktemp("DL2_real")


@pytest.fixture(scope="session")
def temp_DL3(tmp_path_factory):
    return tmp_path_factory.mktemp("DL3")


@pytest.fixture(scope="session")
def temp_pandas(tmp_path_factory):
    return tmp_path_factory.mktemp("pandas")


@pytest.fixture(scope="session")
def temp_DL1_M_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_MAGIC_monly")


@pytest.fixture(scope="session")
def temp_DL1_M_merge_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL1_MAGIC_merge_monly")


@pytest.fixture(scope="session")
def temp_stereo_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("stereo_monly")


@pytest.fixture(scope="session")
def temp_DL2_real_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL2_real_monly")


@pytest.fixture(scope="session")
def temp_DL3_monly(tmp_path_factory):
    return tmp_path_factory.mktemp("DL3_monly")


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
    return "http://www.magic.iac.es/mcp-testdata"


@pytest.fixture(scope="session")
def env_prefix():
    # ENVIRONMENT VARIABLES TO BE CREATED
    return "MAGIC_CTA_DATA_"


"""
Downloads: files
"""


@pytest.fixture(scope="session")
def dl0_gamma(base_url, env_prefix):
    gamma_dl0 = []
    for file in DL0_gamma_data:
        download_path = download_file_cached(
            name=f"DL0gamma/{file}",
            cache_name="magicctapipe",
            env_prefix=env_prefix,
            auth=True,
            default_url=base_url,
            progress=True,
        )
        gamma_dl0.append(download_path)
    return gamma_dl0


@pytest.fixture(scope="session")
def dl0_p(base_url, env_prefix):
    p_dl0 = []
    for file in DL0_p_data:
        download_path = download_file_cached(
            name=f"DL0p/{file}",
            cache_name="magicctapipe",
            env_prefix=env_prefix,
            auth=True,
            default_url=base_url,
            progress=True,
        )
        p_dl0.append(download_path)

    return p_dl0


@pytest.fixture(scope="session")
def dl0_m1(base_url, env_prefix):
    MI_dl0 = []
    for file in DL0_M1_data:
        download_path = download_file_cached(
            name=f"MAGIC/{file}",
            cache_name="magicctapipe",
            env_prefix=env_prefix,
            auth=True,
            default_url=base_url,
            progress=True,
        )
        MI_dl0.append(download_path)
    return MI_dl0


@pytest.fixture(scope="session")
def dl0_m2(base_url, env_prefix):
    MII_dl0 = []
    for file in DL0_M2_data:
        download_path = download_file_cached(
            name=f"MAGIC/{file}",
            cache_name="magicctapipe",
            env_prefix=env_prefix,
            auth=True,
            default_url=base_url,
            progress=True,
        )
        MII_dl0.append(download_path)
    return MII_dl0


@pytest.fixture(scope="session")
def dl1_lst(base_url, env_prefix):
    LST_dl1 = []
    for file in DL1_LST_data:
        download_path = download_file_cached(
            name=f"LST/{file}",
            cache_name="magicctapipe",
            env_prefix=env_prefix,
            auth=True,
            default_url=base_url,
            progress=True,
        )
        LST_dl1.append(download_path)
    return LST_dl1


@pytest.fixture(scope="session")
def config(base_url, env_prefix):
    download_path = download_file_cached(
        name="config.yaml",
        cache_name="magicctapipe",
        env_prefix=env_prefix,
        auth=True,
        default_url=base_url,
        progress=True,
    )
    return download_path


@pytest.fixture(scope="session")
def config_monly(base_url, env_prefix):
    download_path = download_file_cached(
        name="config_monly.yaml",
        cache_name="magicctapipe",
        env_prefix=env_prefix,
        auth=True,
        default_url=base_url,
        progress=True,
    )
    return download_path


"""
Data processing
"""


@pytest.fixture(scope="session")
def gamma_l1(temp_DL1_gamma, dl0_gamma, config):
    """
    Produce a DL1 file
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

    return temp_DL1_gamma


@pytest.fixture(scope="session")
def gamma_l1_monly(temp_DL1_gamma_monly, dl0_gamma, config_monly):
    """
    Produce a DL1 file
    """

    for file in dl0_gamma:
        subprocess.run(
            [
                "lst1_magic_mc_dl0_to_dl1",
                f"-i{str(file)}",
                f"-o{str(temp_DL1_gamma_monly)}",
                f"-c{str(config_monly)}",
            ]
        )

    return temp_DL1_gamma_monly


@pytest.fixture(scope="session")
def gamma_stereo(temp_DL1_gamma_train, temp_DL1_gamma_test, gamma_l1, config):
    """
    Produce a DL1 stereo file
    """

    for i, file in enumerate(gamma_l1.glob("*")):
        if i < ntraingamma:
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
def gamma_stereo_monly(
    temp_DL1_gamma_train_monly, temp_DL1_gamma_test_monly, gamma_l1_monly, config_monly
):
    """
    Produce a DL1 stereo file
    """

    for i, file in enumerate(gamma_l1_monly.glob("*")):
        if i < ntraingamma:
            out = temp_DL1_gamma_train_monly
        else:
            out = temp_DL1_gamma_test_monly
        subprocess.run(
            [
                "lst1_magic_stereo_reco",
                f"-i{str(file)}",
                f"-o{str(out)}",
                f"-c{str(config_monly)}",
                "--magic-only",
            ]
        )

    return (temp_DL1_gamma_train_monly, temp_DL1_gamma_test_monly)


@pytest.fixture(scope="session")
def p_l1(temp_DL1_p, dl0_p, config):
    """
    Produce a DL1 file
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
    return temp_DL1_p


@pytest.fixture(scope="session")
def p_l1_monly(temp_DL1_p_monly, dl0_p, config_monly):
    """
    Produce a DL1 file
    """
    for file in dl0_p:
        subprocess.run(
            [
                "lst1_magic_mc_dl0_to_dl1",
                f"-i{str(file)}",
                f"-o{str(temp_DL1_p_monly)}",
                f"-c{str(config_monly)}",
            ]
        )
    return temp_DL1_p_monly


@pytest.fixture(scope="session")
def p_stereo(temp_DL1_p_train, temp_DL1_p_test, p_l1, config):
    """
    Produce a DL1 stereo file
    """

    for i, file in enumerate(p_l1.glob("*")):
        if i < ntrainp:
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
def p_stereo_monly(
    temp_DL1_p_train_monly, temp_DL1_p_test_monly, p_l1_monly, config_monly
):
    """
    Produce a DL1 stereo file
    """

    for i, file in enumerate(p_l1_monly.glob("*")):
        if i < ntrainp:
            out = temp_DL1_p_train_monly
        else:
            out = temp_DL1_p_test_monly
        subprocess.run(
            [
                "lst1_magic_stereo_reco",
                f"-i{str(file)}",
                f"-o{str(out)}",
                f"-c{str(config_monly)}",
                "--magic-only",
            ]
        )
    return (temp_DL1_p_train_monly, temp_DL1_p_test_monly)


@pytest.fixture(scope="session")
def RF(gamma_stereo, p_stereo, temp_rf, config):
    """
    Produce RFs
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
def RF_monly(gamma_stereo_monly, p_stereo_monly, temp_rf_monly, config_monly):
    """
    Produce RFs
    """

    subprocess.run(
        [
            "lst1_magic_train_rfs",
            f"-g{str(gamma_stereo_monly[0])}",
            f"-p{str(p_stereo_monly[0])}",
            f"-o{str(temp_rf_monly)}",
            f"-c{str(config_monly)}",
            "--train-energy",
            "--train-disp",
            "--train-classifier",
            "--use-unsigned",
        ]
    )
    return temp_rf_monly


@pytest.fixture(scope="session")
def gamma_dl2(temp_DL1_gamma_test, RF, temp_DL2_gamma):
    """
    Produce a DL2 file
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

    subprocess.run(
        [
            "merge_hdf_files",
            f"-i{str(temp_DL2_gamma)}",
            f"-o{str(temp_DL2_gamma)}",
        ]
    )
    for file in temp_DL2_gamma.glob("dl2*run???.h5"):
        subprocess.run(
            [
                "rm",
                f"{file}",
            ]
        )

    return temp_DL2_gamma


@pytest.fixture(scope="session")
def gamma_dl2_monly(temp_DL1_gamma_test_monly, RF_monly, temp_DL2_gamma_monly):
    """
    Produce a DL2 file
    """

    for file in temp_DL1_gamma_test_monly.glob("*"):
        subprocess.run(
            [
                "lst1_magic_dl1_stereo_to_dl2",
                f"-d{str(file)}",
                f"-r{str(RF_monly)}",
                f"-o{str(temp_DL2_gamma_monly)}",
            ]
        )

    subprocess.run(
        [
            "merge_hdf_files",
            f"-i{str(temp_DL2_gamma_monly)}",
            f"-o{str(temp_DL2_gamma_monly)}",
        ]
    )
    for file in temp_DL2_gamma_monly.glob("dl2*run???.h5"):
        subprocess.run(
            [
                "rm",
                f"{file}",
            ]
        )
    return temp_DL2_gamma_monly


@pytest.fixture(scope="session")
def IRF(gamma_dl2, config, temp_irf):
    """
    Produce IRFs
    """

    for file in gamma_dl2.glob("*"):
        subprocess.run(
            [
                "lst1_magic_create_irf",
                f"-g{str(file)}",
                f"-o{str(temp_irf)}",
                f"-c{str(config)}",
            ]
        )
    return temp_irf


@pytest.fixture(scope="session")
def IRF_monly(gamma_dl2_monly, config_monly, temp_irf_monly):
    """
    Produce IRFs
    """

    for file in gamma_dl2_monly.glob("*"):
        subprocess.run(
            [
                "lst1_magic_create_irf",
                f"-g{str(file)}",
                f"-o{str(temp_irf_monly)}",
                f"-c{str(config_monly)}",
            ]
        )
    return temp_irf_monly


@pytest.fixture(scope="session")
def p_dl2(temp_DL1_p_test, RF, temp_DL2_p):
    """
    Produce a DL2 file
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
    return temp_DL2_p


@pytest.fixture(scope="session")
def p_dl2_monly(temp_DL1_p_test_monly, RF_monly, temp_DL2_p_monly):
    """
    Produce a DL2 file
    """

    for file in temp_DL1_p_test_monly.glob("*"):
        subprocess.run(
            [
                "lst1_magic_dl1_stereo_to_dl2",
                f"-d{str(file)}",
                f"-r{str(RF_monly)}",
                f"-o{str(temp_DL2_p_monly)}",
            ]
        )
    return temp_DL2_p_monly


@pytest.fixture(scope="session")
def M1_l1(temp_DL1_M, dl0_m1, config):
    """
    Produce a DL1 file
    """

    subprocess.run(
        [
            "magic_calib_to_dl1",
            f"-i{str(dl0_m1[0])}",
            f"-o{str(temp_DL1_M)}",
            f"-c{str(config)}",
            f"-m{int(maxjoint)}",
            "--process-run",
        ]
    )

    return temp_DL1_M


@pytest.fixture(scope="session")
def M1_l1_monly(temp_DL1_M_monly, dl0_m1, config_monly):
    """
    Produce a DL1 file
    """

    subprocess.run(
        [
            "magic_calib_to_dl1",
            f"-i{str(dl0_m1[0])}",
            f"-o{str(temp_DL1_M_monly)}",
            f"-c{str(config_monly)}",
            f"-m{int(maxmonly)}",
            "--process-run",
        ]
    )

    return temp_DL1_M_monly


@pytest.fixture(scope="session")
def M2_l1(temp_DL1_M, dl0_m2, config):
    """
    Produce a DL1 file
    """

    subprocess.run(
        [
            "magic_calib_to_dl1",
            f"-i{str(dl0_m2[0])}",
            f"-o{str(temp_DL1_M)}",
            f"-c{str(config)}",
            f"-m{int(maxjoint)}",
            "--process-run",
        ]
    )

    return temp_DL1_M


@pytest.fixture(scope="session")
def M2_l1_monly(temp_DL1_M_monly, dl0_m2, config_monly):
    """
    Produce a DL1 file
    """

    subprocess.run(
        [
            "magic_calib_to_dl1",
            f"-i{str(dl0_m2[0])}",
            f"-o{str(temp_DL1_M_monly)}",
            f"-c{str(config_monly)}",
            f"-m{int(maxmonly)}",
            "--process-run",
        ]
    )

    return temp_DL1_M_monly


@pytest.fixture(scope="session")
def merge_magic(M2_l1, M1_l1, temp_DL1_M_merge):
    """
    Merge MAGIC runs
    """

    subprocess.run(
        [
            "merge_hdf_files",
            f"-i{str(M2_l1)}",
            f"-o{str(temp_DL1_M_merge)}",
        ]
    )

    return temp_DL1_M_merge


@pytest.fixture(scope="session")
def merge_magic_monly(M2_l1_monly, M1_l1_monly, temp_DL1_M_merge_monly):
    """
    Merge MAGIC runs
    """

    subprocess.run(
        [
            "merge_hdf_files",
            f"-i{str(M2_l1_monly)}",
            f"-o{str(temp_DL1_M_merge_monly)}",
        ]
    )

    return temp_DL1_M_merge_monly


@pytest.fixture(scope="session")
def coincidence(dl1_lst, merge_magic, temp_coinc, config):
    """
    Coincidence
    """

    for file in dl1_lst:
        subprocess.run(
            [
                "lst1_magic_event_coincidence",
                f"-l{str(file)}",
                f"-m{str(merge_magic)}",
                f"-o{str(temp_coinc)}",
                f"-c{str(config)}",
            ]
        )

    return temp_coinc


@pytest.fixture(scope="session")
def coincidence_stereo(coincidence, temp_coinc_stereo, config):
    """
    Produce stereo coincident events
    """

    for file in coincidence.glob("*"):
        subprocess.run(
            [
                "lst1_magic_stereo_reco",
                f"-i{str(file)}",
                f"-o{str(temp_coinc_stereo)}",
                f"-c{str(config)}",
            ]
        )

    return temp_coinc_stereo


@pytest.fixture(scope="session")
def stereo_monly(merge_magic_monly, temp_stereo_monly, config_monly):
    """
    Produce stereo coincident events
    """

    for file in merge_magic_monly.glob("*"):
        subprocess.run(
            [
                "lst1_magic_stereo_reco",
                f"-i{str(file)}",
                f"-o{str(temp_stereo_monly)}",
                f"-c{str(config_monly)}",
                "--magic-only",
            ]
        )

    return temp_stereo_monly


@pytest.fixture(scope="session")
def real_dl2(coincidence_stereo, RF, temp_DL2_real):
    """
    Produce a DL2 file
    """

    for file in coincidence_stereo.glob("*"):
        subprocess.run(
            [
                "lst1_magic_dl1_stereo_to_dl2",
                f"-d{str(file)}",
                f"-r{str(RF)}",
                f"-o{str(temp_DL2_real)}",
            ]
        )
    return temp_DL2_real


@pytest.fixture(scope="session")
def real_dl2_monly(stereo_monly, RF_monly, temp_DL2_real_monly):
    """
    Produce a DL2 file
    """

    for file in stereo_monly.glob("*"):
        subprocess.run(
            [
                "lst1_magic_dl1_stereo_to_dl2",
                f"-d{str(file)}",
                f"-r{str(RF_monly)}",
                f"-o{str(temp_DL2_real_monly)}",
            ]
        )
    return temp_DL2_real_monly


@pytest.fixture(scope="session")
def real_dl3(real_dl2, IRF, temp_DL3, config):
    """
    Produce a DL3 file
    """

    for file in real_dl2.glob("*"):
        subprocess.run(
            [
                "lst1_magic_dl2_to_dl3",
                f"-d{str(file)}",
                f"-i{str(IRF)}",
                f"-o{str(temp_DL3)}",
                f"-c{str(config)}",
            ]
        )
    return temp_DL3


@pytest.fixture(scope="session")
def real_dl3_monly(real_dl2_monly, IRF_monly, temp_DL3_monly, config_monly):
    """
    Produce a DL3 file
    """

    for file in real_dl2_monly.glob("*"):
        subprocess.run(
            [
                "lst1_magic_dl2_to_dl3",
                f"-d{str(file)}",
                f"-i{str(IRF_monly)}",
                f"-o{str(temp_DL3_monly)}",
                f"-c{str(config_monly)}",
            ]
        )
    return temp_DL3_monly


@pytest.fixture(scope="session")
def real_index(real_dl3):
    """
    Produce indexes
    """

    subprocess.run(
        [
            "create_dl3_index_files",
            f"-i{str(real_dl3)}",
        ]
    )
    return temp_DL3


@pytest.fixture(scope="session")
def real_index_monly(real_dl3_monly):
    """
    Produce indexes
    """

    subprocess.run(
        [
            "create_dl3_index_files",
            f"-i{str(real_dl3_monly)}",
        ]
    )
    return temp_DL3_monly
