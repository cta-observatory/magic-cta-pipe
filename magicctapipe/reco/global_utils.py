import numpy as np
import pandas as pd

__all__ = [
    "compute_event_weights",
    "get_weights_mc_dir_class",
    "check_train_test_intersections",
]


def compute_event_weights():
    """Compute event weights for train scripts

    Returns
    -------
    tuple
        - alt_edges
        - intensity_edges
    """
    sin_edges = np.linspace(0, 1, num=51)
    alt_edges = np.lib.scimath.arcsin(sin_edges)
    intensity_edges = np.logspace(1, 5, num=51)
    return alt_edges, intensity_edges


def get_weights_mc_dir_class(mc_data, alt_edges, intensity_edges):
    mc_hist, _, _ = np.histogram2d(
        mc_data["tel_alt"], mc_data["intensity"], bins=[alt_edges, intensity_edges]
    )

    availability_hist = np.clip(mc_hist, 0, 1)

    # --- MC weights ---
    mc_alt_bins = np.digitize(mc_data["tel_alt"], alt_edges) - 1
    mc_intensity_bins = np.digitize(mc_data["intensity"], intensity_edges) - 1

    # Treating the out-of-range events
    mc_alt_bins[mc_alt_bins == len(alt_edges) - 1] = len(alt_edges) - 2
    mc_intensity_bins[mc_intensity_bins == len(intensity_edges) - 1] = (
        len(intensity_edges) - 2
    )

    mc_weights = 1 / mc_hist[mc_alt_bins, mc_intensity_bins]
    mc_weights *= availability_hist[mc_alt_bins, mc_intensity_bins]

    # --- Storing to a data frame ---
    mc_weight_df = pd.DataFrame(data={"event_weight": mc_weights}, index=mc_data.index)

    return mc_weight_df


def check_train_test_intersections(train, test):
    """Function to check if there are same events in train and test samples

    Parameters
    ----------
    train : pd.DataFrame
        train dataframe
    test : pd.DataFrame
        test dataframe

     Returns
    -------
    bool
        test_passed
    """
    test_passed = True
    cols = list(train.columns)
    df_merge = pd.merge(train, test, on=cols, how="inner")
    if df_merge.empty:
        print("PASS: test and train are different")
    else:
        print("********** WARNING **********")
        print("Same entries for test and train")
        test_passed = False
        print(df_merge)
    return test_passed
