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
    e_ = ("ERROR: can not load the configuration file %s\n"
          "Please check that the file exists and is of YAML format\n"
          "Exiting")
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
    e_ = ("ERROR: can not load the configuration file %s\n"
          "Please check that the file exists and is of YAML format\n"
          "Exiting")
    l_ = ("ERROR: the configuration file is missing the %s section.\n"
          "Exiting")
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


def load_dl1_data_stereo(file, cfg):
    """Load dl1 stereo

    Parameters
    ----------
    file : string
        file
    cfg : dict
        dict loaded from config

    Returns
    -------
    pd.Dataframe
        data
    """
    # Hillas
    data_hillas = pd.read_hdf(file, key=f'dl1/hillas_params')
    data_hillas = drop_keys(data_hillas, cfg['classifier_rf']['extra_keys'])
    # Stereo
    data_stereo = pd.read_hdf(file, key=f'dl1/stereo_params')
    data_stereo = drop_keys(data_stereo, cfg['classifier_rf']['extra_keys'])
    data_stereo = drop_keys(data_stereo, cfg['classifier_rf']['extra_st_keys'])
    data = data_hillas.merge(data_stereo, on=['obs_id', 'event_id'])
    # Index
    data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data.sort_index(inplace=True)
    return data


def load_dl1_data_mono(file, label='hillas_params'):
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
    data = pd.read_hdf(file, key=f'dl1/{label}')
    data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
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
        if(extra_key in df.columns):
            df.drop(extra_key, axis=1, inplace=True)
    return df
