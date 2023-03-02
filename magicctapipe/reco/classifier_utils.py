import pandas as pd
import numpy as np
import glob
import os
import sklearn.metrics


from magicctapipe.utils.utils import info_message
from magicctapipe.utils.filedir import (
    load_dl1_data_stereo_list,
    load_dl1_data_stereo_list_selected,
)
from magicctapipe.reco.global_utils import check_train_test_intersections

__all__ = [
    "GetHist_classifier",
    "evaluate_performance_classifier",
    "get_weights_classifier",
    "print_par_imp_classifier",
    "load_init_data_classifier",
    "check_train_test_intersections_classifier",
]


def GetHist_classifier(data, bins=30, range=None, weights=None):
    hs, edges = np.histogram(data, bins=bins, range=range, weights=weights)
    loc = (edges[1:] + edges[:-1]) / 2

    hist = {}
    hist["Hist"] = hs
    hist["X"] = loc
    hist["XEdges"] = edges

    return hist


def evaluate_performance_classifier(data, class0_name="event_class_0", drop_na=True):
    if drop_na:
        data = data.dropna()

    report = {"gammaness": dict(), "metrics": dict()}

    for event_class in data["true_event_class"].unique():
        events = data.query(f"true_event_class == {event_class}")
        hist = GetHist_classifier(events[class0_name], bins=100, range=(0, 1))
        hist["Hist"] = hist["Hist"] / hist["Hist"].sum()
        hist["Cumsum"] = 1 - np.cumsum(hist["Hist"])

        report["gammaness"][event_class] = hist

    if "mean" in class0_name:
        class_names = list(
            filter(
                lambda name: "event_class_" in name and "_mean" in name, data.columns
            )
        )
    else:
        class_names = list(
            filter(
                lambda name: "event_class_" in name and "_mean" not in name,
                data.columns,
            )
        )

    proba = data[class_names].values
    predicted_class = proba.argmax(axis=1)
    print(data)
    print(class_names)
    print(proba)

    report["metrics"]["acc"] = sklearn.metrics.accuracy_score(
        data["true_event_class"], predicted_class
    )

    true_class = np.clip(data["true_event_class"], 0, 1)
    true_class = 1 - true_class

    try:
        report["metrics"]["auc_roc"] = sklearn.metrics.roc_auc_score(
            true_class, proba[:, 0]
        )
    except Exception as e:
        print(f"ERROR: {e}")

    return report


def get_weights_classifier(mc_data, bkg_data, alt_edges, intensity_edges):
    mc_hist, _, _ = np.histogram2d(
        mc_data["tel_alt"], mc_data["intensity"], bins=[alt_edges, intensity_edges]
    )
    bkg_hist, _, _ = np.histogram2d(
        bkg_data["tel_alt"], bkg_data["intensity"], bins=[alt_edges, intensity_edges]
    )

    availability_hist = np.clip(mc_hist, 0, 1) * np.clip(bkg_hist, 0, 1)

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

    # --- Bkg weights ---
    bkg_alt_bins = np.digitize(bkg_data["tel_alt"], alt_edges) - 1
    bkg_intensity_bins = np.digitize(bkg_data["intensity"], intensity_edges) - 1

    # Treating the out-of-range events
    bkg_alt_bins[bkg_alt_bins == len(alt_edges) - 1] = len(alt_edges) - 2
    bkg_intensity_bins[bkg_intensity_bins == len(intensity_edges) - 1] = (
        len(intensity_edges) - 2
    )

    bkg_weights = 1 / bkg_hist[bkg_alt_bins, bkg_intensity_bins]
    bkg_weights *= availability_hist[bkg_alt_bins, bkg_intensity_bins]

    # --- Storing to a data frame ---
    mc_weight_df = pd.DataFrame(data={"event_weight": mc_weights}, index=mc_data.index)
    bkg_weight_df = pd.DataFrame(
        data={"event_weight": bkg_weights}, index=bkg_data.index
    )

    return mc_weight_df, bkg_weight_df


def print_par_imp_classifier(class_estimator):
    for tel_id in class_estimator.telescope_classifiers:
        feature_importances = class_estimator.telescope_classifiers[
            tel_id
        ].feature_importances_

        print(f"  tel_id: {tel_id}")
        z_ = zip(class_estimator.feature_names, feature_importances)
        for feature, importance in z_:
            print(f"  {feature:.<15s}: {importance:.4f}")
        print("")


def load_init_data_classifier(cfg, mode="train"):
    """Load and init data for classifier train RFs

    Parameters
    ----------
    cfg : dict
        configurations loaded from configuration file
    mode : str, optional
        train or test, by default "train"

    Returns
    -------
    tuple
        mc_data, bkg_data
    """
    mono_mode = len(cfg["all_tels"]["tel_ids"]) == 1
    f_ = cfg["data_files"]["mc"][f"{mode}_sample"]["hillas_h5"]
    f_ = f"{os.path.dirname(f_)}/*{os.path.splitext(f_)[1]}"
    fl_ = glob.glob(f_)
    info_message(f"Loading MC {mode} data...", prefix="ClassifierRF")
    if mode == "test":
        mc_data = load_dl1_data_stereo_list_selected(
            file_list=fl_,
            sub_dict=cfg["classifier_rf"],
            file_n_key="test_file_n",
            drop=True,
            mono_mode=mono_mode,
        )
    else:
        mc_data = load_dl1_data_stereo_list(fl_, drop=True, mono_mode=mono_mode)

    f_ = cfg["data_files"]["data"][f"{mode}_sample"]["hillas_h5"]
    f_ = f"{os.path.dirname(f_)}/*{os.path.splitext(f_)[1]}"
    fl_ = glob.glob(f_)
    info_message(f'Loading "off" {mode} data...', prefix="ClassifierRF")
    if mode == "test":
        bkg_data = load_dl1_data_stereo_list_selected(
            file_list=fl_,
            sub_dict=cfg["classifier_rf"],
            file_n_key="test_file_n",
            drop=True,
            mono_mode=mono_mode,
        )
    else:
        bkg_data = load_dl1_data_stereo_list(fl_, drop=True, mono_mode=mono_mode)

    # True event classes
    mc_data["true_event_class"] = 0
    bkg_data["true_event_class"] = 1

    # Dropping data with the wrong altitude
    bkg_data = bkg_data.query(cfg["global"]["wrong_alt"])

    return mc_data, bkg_data


def check_train_test_intersections_classifier(
    mc_data_train, bkg_data_train, mc_data_test, bkg_data_test
):
    """Function to check if there are same events in train and test samples, for
    train rfs classifier

    Parameters
    ----------
    mc_data_train : pd.DataFrame
        mc_data_train
    bkg_data_train : pd.DataFrame
        bkg_data_train
    mc_data_test : pd.DataFrame
        mc_data_test
    bkg_data_test : pd.DataFrame
        bkg_data_test

     Returns
    -------
    bool
        test_passed
    """
    tests = {
        "data": [mc_data_train, mc_data_test],
        "bkg": [bkg_data_train, bkg_data_test],
    }
    test_passed = True
    for k in tests.keys():
        print(f"Analizing {k} test and train")
        test_passed_ = check_train_test_intersections(*tests[k])
        test_passed = test_passed and test_passed_
    return test_passed
