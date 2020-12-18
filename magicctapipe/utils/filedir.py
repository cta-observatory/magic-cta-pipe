import os
import sys
import yaml
import datetime
import pandas as pd


def load_cfg_file(config_file):
    """Load the configuration file (yaml format)

    Parameters
    ----------
    config_file : str
        configuration file, yaml format

    Returns
    -------
    dict
        loaded configurations
    """
    e_ = (
        "ERROR: can not load the configuration file %s\n"
        "Please check that the file exists and is of YAML format\n"
        "Exiting"
    )
    try:
        cfg = yaml.safe_load(open(config_file, "r"))
    except IOError:
        print(e_ % config_file)
        sys.exit()
    return cfg


def load_cfg_file_check(config_file, label):
    """Load the configuration file (yaml format) and checks that the label
    section is present in the given file, if not it exits

    Parameters
    ----------
    config_file : str
        configuration file, yaml format
    label : str
        label for the desired section

    Returns
    -------
    dict
        loaded configurations
    """
    e_ = (
        "ERROR: can not load the configuration file %s\n"
        "Please check that the file exists and is of YAML format\n"
        "Exiting"
    )
    l_ = "ERROR: the configuration file is missing the %s section.\n" "Exiting"
    cfg = load_cfg_file(config_file)
    if label not in cfg:
        print(l_ % label)
        sys.exit()
    return cfg


def check_folder(folder):
    """Check if folder exists; if not, it will be created"""
    if not os.path.exists(folder):
        print("Directory %s does not exist, creating it..." % folder)
        os.makedirs(folder)


def load_dl1_data_stereo(file, drop=False):
    """Load dl1 data hillas and stereo and merge them togheter

    Parameters
    ----------
    file : string
        file
    drop : bool, optional
        drop extra keys, by default False


    Returns
    -------
    pd.Dataframe
        data
    """
    extra_keys = [
        "true_energy",
        "true_alt",
        "true_az",
        "mjd",
        "goodness_of_fit",
        "h_max_uncert",
        "az_uncert",
        "core_uncert",
    ]
    extra_stereo_keys = [
        "true_energy",
        "true_alt",
        "true_az",
        "tel_alt",
        "tel_az",
        "num_islands",
        "n_islands",
        "tel_id",
    ]
    # Hillas
    data_hillas = pd.read_hdf(file, key=f"dl1/hillas_params")
    # Stereo
    data_stereo = pd.read_hdf(file, key=f"dl1/stereo_params")
    # Drop common keys
    data_stereo = drop_keys(data_stereo, extra_stereo_keys)
    # Drop extra keys
    if drop:
        data_hillas = drop_keys(data_hillas, extra_keys)
        data_stereo = drop_keys(data_stereo, extra_keys)
    # Merge
    data = data_hillas.merge(data_stereo, on=["obs_id", "event_id"])
    # Index
    data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    data.sort_index(inplace=True)
    return data


def load_dl1_data_mono(file, label="hillas_params"):
    """Load `dl1/{label}` from dl1 file, h5 format for mono data

    Parameters
    ----------
    file : str
        file name
    label : str, optional
        dl1 label, by default 'hillas_params'

    Returns
    -------
    pandas.DataFrame
        data
    """
    data = pd.read_hdf(file, key=f"dl1/{label}")
    data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    data.sort_index(inplace=True)
    return data


def drop_keys(df, extra_keys):
    """Drop extrakeys from pandas dataframe, without crashing if they are not
    present in the dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe
    extra_keys : list
        list of keys to be dropped

    Returns
    -------
    pandas.DataFrame
        dataframe without extra keys
    """
    for extra_key in extra_keys:
        if extra_key in df.columns:
            df.drop(extra_key, axis=1, inplace=True)
    return df
