import os
import sys
import glob
import yaml
import datetime
import pandas as pd


def load_cfg_file(config_file):
    """Loads the configuration file (yaml format)

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
    """Loads the configuration file (yaml format) and checks that the label
    section is present in the given file, otherwise it exits

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
    """Checks if folder exists; if not, it will be created

    Parameters
    ----------
    folder : str
        folder name (with path)
    """
    if not os.path.exists(folder):
        print("Directory %s does not exist, creating it..." % folder)
        os.makedirs(folder)


def load_dl1_data_stereo_list_selected(
    file_list, sub_dict, file_n_key="file_n", drop=False
):
    """Loads dl1 data hillas and stereo and merge them togheter, from `file_list`. 
    If in `sub_dict` it finds the `file_n_key` key, and the given number is > 0, it 
    limits the `file_list` lenght to the given number. Useful to make random forests
    plot on a small test sample

    Parameters
    ----------
    file_list : string
        file_list
    sub_dict : dict
        sub-dictionary loaded from config file (e.g. cfg["direction_rf"])
    file_n_key : str, optional
        file number key, by default "file_n"
    drop : bool, optional
        drop extra keys, by default False

    Returns
    -------
    pd.Dataframe
        data
    """
    if file_n_key in sub_dict.keys():
        n = sub_dict[file_n_key]
        if n > 0:
            data = load_dl1_data_stereo_list(file_list[:n], drop)
    else:
        data = load_dl1_data_stereo_list(file_list, drop)
    return data


def load_dl1_data_stereo_list(file_list, drop=False):
    """Loads dl1 data hillas and stereo and merge them togheter, from `file_list`

    Parameters
    ----------
    file_list : string
        file_list
    drop : bool, optional
        drop extra keys, by default False


    Returns
    -------
    pd.Dataframe
        data
    """
    for i, file in enumerate(file_list):
        data_ = load_dl1_data_stereo(file, drop)
        if i == 0:
            data = data_
        else:
            data = data.append(data_)
    return data


def load_dl1_data_stereo(file, drop=False):
    """Loads dl1 data (hillas and stereo) from `file` and merge them togheter

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
    """Loads `dl1/{label}` from dl1 file, h5 format for mono data

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
    """Drops extrakeys from pandas dataframe, without crashing if they are not
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


def out_file_h5_no_run(in_file, li, hi):
    """Returns the h5 output file name, from a simtel.gz input file, without run number

    Parameters
    ----------
    in_file : str
        Input file
    li : int
        low index
    hi : int
        high index

    Returns
    -------
    str
        h5 output file, absolute path
    """
    f = os.path.basename(in_file)
    out = "_".join(f.split("_")[:li] + f.split("_")[hi:])
    out = "%s.h5" % out.rstrip(".simtel.gz")
    out = os.path.join(os.path.dirname(in_file), out)
    return out


def out_file_h5(in_file):
    """Returns the h5 output file name, from a simtel.gz input file. Only file name,
    without path

    Parameters
    ----------
    in_file : str
        Input file

    Returns
    -------
    str
        h5 output file, without path
    """
    f = os.path.basename(in_file)
    out = "%s.h5" % f.rstrip(".simtel.gz")
    return out
