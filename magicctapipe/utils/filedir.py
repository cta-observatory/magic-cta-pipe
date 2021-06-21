import os
import sys
import glob
import yaml
import copy
import datetime
import pandas as pd
import numpy as np


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
    print(f"Loading configuration file\n{config_file}")
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
        try:
            os.makedirs(folder)
        except Exception as e:
            print(f"ERROR, folder not created. {e}")


def load_dl1_data_stereo_list_selected(
    file_list, sub_dict, file_n_key="file_n", drop=False, mono_mode=False
):
    """Loads dl1 data hillas and stereo and merge them togheter, from `file_list`. 
    If in `sub_dict` it finds the `file_n_key` key, and if the given number is > 0, it 
    limits the `file_list` lenght to the given number. Useful to make random forests
    plot on a smaller test sample

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
    mono_mode : bool, optional
        mono mode, by default False

    Returns
    -------
    pd.Dataframe
        data
    """
    if file_n_key in sub_dict.keys():
        n = sub_dict[file_n_key]
        if n > 0:
            file_list = file_list[:n]
    data = load_dl1_data_stereo_list(file_list, drop, mono_mode=mono_mode, verbose=True)
    return data


def load_dl1_data_stereo_list(file_list, drop=False, verbose=False, mono_mode=False):
    """Loads dl1 data hillas and stereo and merge them togheter, from `file_list`

    Parameters
    ----------
    file_list : string
        file_list
    drop : bool, optional
        drop extra keys, by default False
    verbose : bool, optional
        print file list, by default False
    mono_mode : bool, optional
        mono mode, by default False


    Returns
    -------
    pd.Dataframe
        data
    """
    if verbose:
        fl = "\n".join(file_list)
        print(f"File list:\n{fl}")
    first_time = True
    for i, file in enumerate(file_list):
        try:
            if not mono_mode:
                data_ = load_dl1_data_stereo(file, drop)
            else:
                data_ = load_dl1_data_mono(file)
        except Exception as e:
            print(f"LOAD FAILED with file {file}", e)
            continue
        if first_time:
            data = data_
            first_time = False
        else:
            data = data.append(data_)
    return data


def load_dl1_data_stereo(file, drop=False, slope_abs=True):
    """Loads dl1 data (hillas and stereo) from `file` and merge them togheter

    Parameters
    ----------
    file : string
        file
    drop : bool, optional
        drop extra keys, by default False
    slope_abs : bool, optional
        get only the absolute value of slope, by default True


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
        "true_core_x",
        "true_core_y",
    ]
    extra_stereo_keys = ["tel_alt", "tel_az", "num_islands", "n_islands", "tel_id"]
    common_keys = ["obs_id", "event_id", "true_energy", "true_alt", "true_az"]
    try:
        # Hillas
        data_hillas = pd.read_hdf(file, key=f"dl1/hillas_params")
        # Stereo
        data_stereo = pd.read_hdf(file, key=f"dl1/stereo_params")
        # Drop extra stereo keys
        data_stereo = drop_keys(data_stereo, extra_stereo_keys)
        # Drop extra keys
        if drop:
            data_hillas = drop_keys(data_hillas, extra_keys)
            data_stereo = drop_keys(data_stereo, extra_keys)
        # Check common keys
        common_keys = check_common_keys(data_hillas, data_stereo, common_keys)
        # Merge
        data = data_hillas.merge(data_stereo, on=common_keys)
        # Index
        data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    except:
        data = pd.read_hdf(file, key=f"dl2/reco")
    data.sort_index(inplace=True)
    # Get absolute value of slope, if needed
    if slope_abs:
        data["slope"] = abs(data["slope"])
    return data


def load_dl1_data_mono(file, label="hillas_params", slope_abs=True):
    """Loads `dl1/{label}` from dl1 file, h5 format for mono data

    Parameters
    ----------
    file : str
        file name
    label : str, optional
        dl1 label, by default 'hillas_params'
    slope_abs : bool, optional
        get only the absolute value of slope, by default True

    Returns
    -------
    pandas.DataFrame
        data
    """
    data = pd.read_hdf(file, key=f"dl1/{label}")
    data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    data.sort_index(inplace=True)
    # Get absolute value of slope, if needed
    if slope_abs:
        data["slope"] = abs(data["slope"])
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


def check_common_keys(df1, df2, common_keys):
    """Check if `common_keys` exist both in `df1` and in `df2`, and return sub-list 
    of `common_keys` which really exist in both dataframe

    Parameters
    ----------
    df1 : pandas.DataFrame
        dataframe
    df2 : pandas.DataFrame
        dataframe
    common_keys : list
        list of common keys to be checked

    Returns
    -------
    list
        list of common keys
    """
    common_keys_checked = []
    for k in common_keys:
        if k in df1.columns and k in df2.columns:
            common_keys_checked.append(k)
    return common_keys_checked


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


def out_file_h5_reco(in_file):
    """Returns the h5 reco output file name, from a h5 dl1 input file. Only file name,
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
    out = "%s_reco.h5" % f.rstrip(".h5")
    return out


def read_mc_header(file, mc_header_key="dl2/mc_header"):
    """Function to read `mc_header` from DL2 file

    Parameters
    ----------
    file : str
        file
    mc_header_key : str, optional
        mc_header key, by default "dl2/mc_header"

    Returns
    -------
    pd.DataFrame
        mc_header
    """
    return pd.read_hdf(file, key=mc_header_key)


def save_yaml_np(data, file):
    """Save dictionary to yaml file, converting `np.array` objects to `list`

    Parameters
    ----------
    data : dict
        data to be saved
    file : str
        yaml file name
    """
    with open(file, "w") as f_:
        yaml.dump(convert_np_list_dict(copy.deepcopy(data)), f_)


def convert_np_list_dict(d):
    """Loop on dictionary and convert `np.array` objects to list

    Parameters
    ----------
    d : dict
        input dictionary

    Returns
    -------
    dict
        output dictionary
    """
    for k in d.keys():
        if type(d[k]) == dict:
            convert_np_list_dict(d[k])
        elif type(d[k]) == np.ndarray:
            d[k] = d[k].tolist()
    return d

