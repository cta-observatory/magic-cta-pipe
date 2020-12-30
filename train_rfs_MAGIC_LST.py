# coding: utf-8

import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from magicctapipe.utils.plot import *
from magicctapipe.utils.utils import *
from magicctapipe.utils.tels import *
from magicctapipe.utils.filedir import *
from magicctapipe.train.global_utils import *
from magicctapipe.train.classifier_utils import *
from magicctapipe.train.direction_utils import *
from magicctapipe.train.energy_utils import *
from magicctapipe.train.event_processing import EventClassifierPandas
from magicctapipe.train.event_processing import EnergyEstimatorPandas
from magicctapipe.train.event_processing import DirectionEstimatorPandas

PARSER = argparse.ArgumentParser(
    description="Trains random forests for stereo data",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
PARSER.add_argument(
    "-cfg",
    "--config_file",
    type=str,
    required=True,
    help="Configuration file, yaml format",
)
PARSER.add_argument(
    "-c",
    "--classifier",
    action="store_true",
    required=False,
    default=False,
    help="Train classifier random forests",
)
PARSER.add_argument(
    "-e",
    "--energy",
    action="store_true",
    required=False,
    default=False,
    help="Train energy random forests",
)
PARSER.add_argument(
    "-d",
    "--direction",
    action="store_true",
    required=False,
    default=False,
    help="Train direction random forests",
)
PARSER.add_argument(
    "-a",
    "--all",
    action="store_true",
    required=False,
    default=False,
    help="Train all random forests",
)


def train_classifier_rf_stereo(config_file):
    print_title("TRAIN CLASSIFIER RFs")

    # --- Reading the configuration file ---
    cfg = load_cfg_file_check(config_file=config_file, label="classifier_rf")

    # --- Check output directory ---
    check_folder(cfg["classifier_rf"]["save_dir"])

    # --- Train sample ---
    mc_data_train, bkg_data_train = load_init_data_classifier(mode="train", cfg=cfg)

    # --- Test sample ---
    mc_data_test, bkg_data_test = load_init_data_classifier(mode="test", cfg=cfg)

    # --- Check intersections ---
    # useful ONLY if test_file_n == 0
    wn_ = "WARNING: test_file_n != 0, considering only a selection of the test sample"
    if "check_train_test" in cfg["classifier_rf"].keys():
        if cfg["classifier_rf"]["check_train_test"]:
            info_message("Check train and test", prefix="ClassifierRF")
            if cfg["classifier_rf"]["test_file_n"] > 0:
                info_message(wn_, prefix="ClassifierRF")
            test_passed = check_train_test_intersections_classifier(
                mc_data_train=mc_data_train,
                bkg_data_train=bkg_data_train,
                mc_data_test=mc_data_test,
                bkg_data_test=bkg_data_test,
            )
            s_ = "Test PASSED" if test_passed else "Test NOT PASSED"
            info_message(s_, prefix="ClassifierRF")

    # Computing event weights
    alt_edges, intensity_edges = compute_event_weights()

    mc_weights, bkg_weights = get_weights_classifier(
        mc_data_train, bkg_data_train, alt_edges, intensity_edges
    )

    mc_data_train = mc_data_train.join(mc_weights)
    bkg_data_train = bkg_data_train.join(bkg_weights)

    # Merging the train sample
    shower_data_train = mc_data_train.append(bkg_data_train)

    # Merging the test sample
    shower_data_test = mc_data_test.append(bkg_data_test)

    info_message("Preprocessing...", prefix="ClassifierRF")

    # --- Data preparation ---
    l_ = ["obs_id", "event_id"]
    shower_data_train["multiplicity"] = (
        shower_data_train["intensity"].groupby(level=l_).count()
    )
    shower_data_test["multiplicity"] = (
        shower_data_test["intensity"].groupby(level=l_).count()
    )

    # Applying the cuts
    c_ = cfg["classifier_rf"]["cuts"]
    shower_data_train = shower_data_train.query(c_)
    shower_data_test = shower_data_test.query(c_)

    # --- Training the classifier RF ---
    info_message("Training RF...", prefix="ClassifierRF")

    class_estimator = EventClassifierPandas(
        cfg["classifier_rf"]["features"], **cfg["classifier_rf"]["settings"]
    )
    class_estimator.fit(shower_data_train)

    # --- Save RF data to joblib file ---
    class_estimator.save(
        os.path.join(
            cfg["classifier_rf"]["save_dir"], cfg["classifier_rf"]["joblib_name"]
        )
    )

    # --- Show results ---
    # Print Parameter importances Mono
    info_message("Parameter importances", prefix="ClassifierRF")
    print_par_imp_classifier(class_estimator)

    # Apply RF
    info_message("Applying RF...", prefix="ClassifierRF")
    # Mono
    class_reco = class_estimator.predict(shower_data_test)
    shower_data_test = shower_data_test.join(class_reco)

    # Evaluating performance
    info_message("Evaluating performance...", prefix="ClassifierRF")

    idx = pd.IndexSlice

    performance = dict()
    # tel_ids = shower_data_test.index.levels[2]

    tel_ids, tel_ids_LST, tel_ids_MAGIC = check_tel_ids(cfg)

    # Mean
    performance[0] = evaluate_performance_classifier(
        shower_data_test.loc[idx[:, :, tel_ids[0]], shower_data_test.columns],
        class0_name="event_class_0_mean",
    )

    # For tels
    for tel_id in tel_ids:
        performance[tel_id] = evaluate_performance_classifier(
            shower_data_test.loc[idx[:, :, tel_id], shower_data_test.columns]
        )

    # ================
    # === Plotting ===
    # ================

    plt.figure(figsize=tuple(cfg["classifier_rf"]["fig_size"]))
    labels = ["Gamma", "Hadrons"]

    grid_shape = (2, len(tel_ids) + 1)

    for tel_num, tel_id in enumerate(performance):
        plt.subplot2grid(grid_shape, (0, tel_num))
        if tel_id == 0:
            plt.title("Mean")
        else:
            n_ = get_tel_name(tel_id=tel_id, cfg=cfg)
            plt.title(f"{n_} estimation")
        plt.xlabel("Gammaness")
        # plt.xlabel('Class 0 probability')
        plt.ylabel("Event density")

        gammaness = performance[tel_id]["gammaness"]
        print(performance[tel_id]["metrics"])

        for class_i, event_class in enumerate(gammaness):
            plt.step(
                gammaness[event_class]["XEdges"][:-1],
                gammaness[event_class]["Hist"],
                where="post",
                color=f"C{class_i}",
                label=labels[class_i],
            )
            #  label=f'Class {event_class}')

            plt.step(
                gammaness[event_class]["XEdges"][1:],
                gammaness[event_class]["Hist"],
                where="pre",
                color=f"C{class_i}",
            )

            plt.fill_between(
                gammaness[event_class]["XEdges"][:-1],
                gammaness[event_class]["Hist"],
                step="post",
                color=f"C{class_i}",
                alpha=0.3,
            )

            value = performance[tel_id]["metrics"]["acc"]
            plt.text(
                0.9,
                0.9,
                f"acc={value:.2f}",
                ha="right",
                va="top",
                transform=plt.gca().transAxes,
            )

            value = performance[tel_id]["metrics"]["auc_roc"]
            plt.text(
                0.9,
                0.8,
                f"auc_roc={value:.2f}",
                ha="right",
                va="top",
                transform=plt.gca().transAxes,
            )

        plt.legend()

    for tel_num, tel_id in enumerate(performance):
        plt.subplot2grid(grid_shape, (1, tel_num))
        plt.semilogy()
        if tel_id == 0:
            # plt.title(f'Tel {tel_id} estimation')
            plt.title("Mean")
        elif tel_id == -1:
            plt.title("Stereo")
        else:
            n_ = get_tel_name(tel_id=tel_id, cfg=cfg)
            plt.title(f"{n_} estimation")
        plt.xlabel("Gammaness")
        # plt.xlabel('Class 0 probability')
        plt.ylabel("Cumulative probability")
        plt.ylim(1e-3, 1)

        gammaness = performance[tel_id]["gammaness"]

        for class_i, event_class in enumerate(gammaness):
            plt.step(
                gammaness[event_class]["XEdges"][:-1],
                gammaness[event_class]["Cumsum"],
                where="post",
                color=f"C{class_i}",
                label=labels[class_i],
            )
            #  label=f'Class {event_class}')

            plt.step(
                gammaness[event_class]["XEdges"][1:],
                gammaness[event_class]["Cumsum"],
                where="pre",
                color=f"C{class_i}",
            )

            plt.fill_between(
                gammaness[event_class]["XEdges"][:-1],
                gammaness[event_class]["Cumsum"],
                step="post",
                color=f"C{class_i}",
                alpha=0.3,
            )

        plt.legend()

    plt.tight_layout()
    save_plt(
        n=cfg["classifier_rf"]["fig_name"],
        rdir=cfg["classifier_rf"]["save_dir"],
        vect="pdf",
    )

    plt.close()


def train_direction_rf_stereo(config_file):
    print_title("TRAIN DIRECTION RFs")

    # Load config_file
    cfg = load_cfg_file_check(config_file=config_file, label="direction_rf")

    # --- Check output directory ---
    check_folder(cfg["direction_rf"]["save_dir"])

    # --- Train sample ---
    info_message("Loading train data...", prefix="DirRF")
    f_ = cfg["data_files"]["mc"]["train_sample"]["hillas_h5"]
    info_message(f"Loading train files with the following mask:\n{f_}", prefix="DirRF")
    shower_data_train = load_dl1_data_stereo_list(glob.glob(f_))

    # --- Test sample ---
    f_ = cfg["data_files"]["mc"]["test_sample"]["hillas_h5"]
    info_message(f"Loading test files with the following mask:\n{f_}", prefix="DirRF")
    shower_data_test = load_dl1_data_stereo_list_selected(
        file_list=glob.glob(f_), sub_dict=cfg["direction_rf"], file_n_key="test_file_n"
    )

    # --- Check intersections ---
    wt_ = "WARNING: check only on gammas; use it in classifier to check also protons"
    # useful ONLY if test_file_n == 0
    wn_ = "WARNING: test_file_n != 0, considering only a selection of the test sample"
    if "check_train_test" in cfg["direction_rf"].keys():
        if cfg["direction_rf"]["check_train_test"]:
            info_message("Check train and test", prefix="DirRF")
            info_message(wt_, prefix="DirRF")
            if cfg["direction_rf"]["test_file_n"] > 0:
                info_message(wn_, prefix="DirRF")
            test_passed = check_train_test_intersections(
                shower_data_train, shower_data_test
            )
            s_ = "Test PASSED" if test_passed else "Test NOT PASSED"
            info_message(s_, prefix="DirRF")

    # Computing event weights
    info_message("Computing the train sample event weights...", prefix="DirRF")
    alt_edges, intensity_edges = compute_event_weights()

    mc_weights = get_weights_mc_dir_class(shower_data_train, alt_edges, intensity_edges)

    shower_data_train = shower_data_train.join(mc_weights)

    tel_ids, tel_ids_LST, tel_ids_MAGIC = check_tel_ids(cfg)

    # --- Data preparation ---
    l_ = ["obs_id", "event_id"]
    shower_data_train["multiplicity"] = (
        shower_data_train["intensity"].groupby(level=l_).count()
    )
    shower_data_test["multiplicity"] = (
        shower_data_test["intensity"].groupby(level=l_).count()
    )

    # Applying the cuts
    shower_data_train = shower_data_train.query(cfg["direction_rf"]["cuts"])
    shower_data_test = shower_data_test.query(cfg["direction_rf"]["cuts"])

    # --- MAGIC - LST description ---
    array_tel_descriptions = get_array_tel_descriptions(
        tel_ids_LST=tel_ids_LST, tel_ids_MAGIC=tel_ids_MAGIC
    )

    # --- Training the direction RF ---
    info_message("Training the RF\n", prefix="DirRF")

    direction_estimator = DirectionEstimatorPandas(
        cfg["direction_rf"]["features"],
        array_tel_descriptions,
        **cfg["direction_rf"]["settings"],
    )
    direction_estimator.fit(shower_data_train)
    direction_estimator.save(
        os.path.join(
            cfg["direction_rf"]["save_dir"], cfg["direction_rf"]["joblib_name"]
        )
    )

    # Printing the parameter "importances"
    for kind in direction_estimator.telescope_rfs:
        rfs_ = direction_estimator.telescope_rfs[kind]
        for tel_id in rfs_:
            feature_importances = rfs_[tel_id].feature_importances_
            print(f"  Kind: {kind}, tel_id: {tel_id}")
            z = zip(cfg["direction_rf"]["features"][kind], feature_importances)
            for feature, importance in z:
                print(f"  {feature:.<15s}: {importance:.4f}")
            print("")

    # --- Applying RF to the "test" sample ---
    info_message('Applying RF to the "test" sample', prefix="DirRF")
    coords_reco = direction_estimator.predict(shower_data_test)
    shower_data_test = shower_data_test.join(coords_reco)

    # --- Evaluating the performance ---
    info_message("Evaluating the performance", prefix="DirRF")
    separation_df = compute_separation_angle_direction(shower_data_test)

    # Energy-dependent resolution
    info_message("Estimating the energy-dependent resolution", prefix="DirRF")
    energy_edges = np.logspace(-1, 1.3, num=20)
    energy = (energy_edges[1:] * energy_edges[:-1]) ** 0.5

    energy_psf = dict()
    # for i in range(3):
    for i in [0] + tel_ids:
        energy_psf[i] = np.zeros_like(energy)

    for ei in range(len(energy_edges) - 1):
        cuts = f"(true_energy>= {energy_edges[ei]:.2e})"
        cuts += f" & (true_energy < {energy_edges[ei+1]: .2e})"
        # cuts += ' & (intensity > 100)'
        # cuts += ' & (length > 0.05)'
        cuts += " & (multiplicity > 1)"
        query = separation_df.query(cuts)

        # for pi in range(3): # OLD
        for pi in [0] + tel_ids:
            if pi > 0:
                tel_id = pi
            else:
                tel_id = 1
            try:
                selection = query.loc[
                    (slice(None), slice(None), tel_id), f"sep_{pi}"
                ].dropna()
                energy_psf[pi][ei] = np.percentile(selection, 68)
            except Exception as e:
                print(f"ERROR: {e}. Setting energy_psf to 0")
                energy_psf[pi][ei] = 0

    # Offset-dependent resolution
    info_message("Estimating the offset-dependent resolution", prefix="DirRF")
    offset = angular_separation(
        separation_df["tel_az"],
        separation_df["tel_alt"],
        separation_df["true_az"],
        separation_df["true_alt"],
    )

    separation_df["offset"] = np.degrees(offset)

    offset_edges = np.linspace(0, 1.3, num=10)
    offset = (offset_edges[1:] * offset_edges[:-1]) ** 0.5

    offset_psf = dict()
    # for i in range(3): # OLD
    for i in [0] + tel_ids:
        offset_psf[i] = np.zeros_like(offset)

    for oi in range(len(offset_edges) - 1):
        cuts = f"(offset >= {offset_edges[oi]:.2f})"
        cuts += f" & (offset < {offset_edges[oi+1]:.2f})"
        # cuts += ' & (intensity > 100)'
        # cuts += ' & (length > 0.05)'
        cuts += " & (multiplicity > 1)"
        query = separation_df.query(cuts)

        # for pi in range(3):
        for pi in [0] + tel_ids:
            if pi > 0:
                tel_id = pi
            else:
                tel_id = 1
            try:
                selection = query.loc[
                    (slice(None), slice(None), tel_id), [f"sep_{pi}"]
                ].dropna()
                offset_psf[pi][oi] = np.percentile(selection[f"sep_{pi}"], 68)
            except Exception as e:
                print(f"ERROR: {e}. Setting offset_psf to 0")
                offset_psf[pi][oi] = 0

    # ================
    # === Plotting ===
    # ================

    plt.figure(figsize=tuple(cfg["direction_rf"]["fig_size"]))
    # plt.style.use('presentation')

    plt.xlabel(r"$\theta^2$, deg$^2$")

    # for tel_id in [0, 1, 2]:
    grid_shape = (len(tel_ids) + 1, 2)
    for index, tel_id in enumerate([0] + tel_ids):
        plt.subplot2grid(grid_shape, (index, 0))
        if tel_id == 0:
            plt.title(f"Total")
        else:
            plt.title(get_tel_name(tel_id=tel_id, cfg=cfg))
        plt.xlabel(r"$\theta^2$, deg$^2$")
        # plt.semilogy()
        plt.hist(
            separation_df[f"sep_{tel_id}"] ** 2,
            bins=100,
            range=(0, 0.5),
            density=True,
            alpha=0.1,
            color="C0",
        )
        plt.hist(
            separation_df[f"sep_{tel_id}"] ** 2,
            bins=100,
            range=(0, 0.5),
            density=True,
            histtype="step",
            color="C0",
        )
        plt.grid(linestyle=":")

        plt.subplot2grid(grid_shape, (index, 1))
        plt.xlabel(r"$\theta$, deg")
        plt.xlim(0, 2.0)
        plt.hist(
            separation_df[f"sep_{tel_id}"],
            bins=400,
            range=(0, 5),
            cumulative=True,
            density=True,
            alpha=0.1,
            color="C0",
        )
        plt.hist(
            separation_df[f"sep_{tel_id}"],
            bins=400,
            range=(0, 5),
            cumulative=True,
            density=True,
            histtype="step",
            color="C0",
        )
        plt.grid(linestyle=":")

    plt.tight_layout()
    save_plt(
        n=cfg["direction_rf"]["fig_name_theta2"],
        rdir=cfg["direction_rf"]["save_dir"],
        vect="pdf",
    )
    plt.close()

    plt.clf()

    plt.semilogx()
    plt.xlabel("Energy [TeV]")
    plt.ylabel(r"$\sigma_{68}$ [deg]")
    plt.ylim(0, 1.0)
    plt.plot(energy, energy_psf[0], linewidth=4, label="Total")
    for tel_id in tel_ids:
        for i, tel_label in enumerate(cfg["all_tels"]["tel_n"]):
            if tel_id in cfg[tel_label]["tel_ids"]:
                l_ = get_tel_name(tel_id=tel_id, cfg=cfg)
                plt.plot(energy, energy_psf[tel_id], label=l_)
    plt.grid(linestyle=":")
    plt.legend()
    save_plt(
        n=cfg["direction_rf"]["fig_name_PSF_energy"],
        rdir=cfg["direction_rf"]["save_dir"],
        vect="pdf",
    )
    plt.close()

    plt.clf()
    plt.xlabel("Offset [deg]")
    plt.ylabel(r"$\sigma_{68}$ [deg]")
    plt.ylim(0, 0.5)
    plt.plot(offset, offset_psf[0], linewidth=4, label="Total")
    for tel_id in tel_ids:
        for i, tel_label in enumerate(cfg["all_tels"]["tel_n"]):
            if tel_id in cfg[tel_label]["tel_ids"]:
                l_ = get_tel_name(tel_id=tel_id, cfg=cfg)
                plt.plot(offset, offset_psf[tel_id], label=l_)
    plt.grid(linestyle=":")
    plt.legend()
    save_plt(
        n=cfg["direction_rf"]["fig_name_PSF_offset"],
        rdir=cfg["direction_rf"]["save_dir"],
        vect="pdf",
    )
    plt.close()


def train_energy_rf_stereo(config_file):
    print_title("TRAIN ENERGY RFs")

    # --- Reading the configuration file ---
    cfg = load_cfg_file_check(config_file=config_file, label="energy_rf")

    # --- Check output directory ---
    check_folder(cfg["energy_rf"]["save_dir"])

    # --- Train sample ---
    f_ = cfg["data_files"]["mc"]["train_sample"]["hillas_h5"]
    info_message("Loading train data...", prefix="EnergyRF")
    info_message(
        f"Loading train data with the following mask: \n{f_}", prefix="EnergyRF"
    )
    info_message(f"Loading files with the following mask:\n{f_}", prefix="EnergyRF")
    shower_data_train = load_dl1_data_stereo_list(glob.glob(f_))

    # --- Test sample ---
    f_ = cfg["data_files"]["mc"]["test_sample"]["hillas_h5"]
    info_message(f"Loading test data with the following mask:\n{f_}", prefix="EnergyRF")
    # shower_data_test = load_dl1_data_stereo_list(glob.glob(f_))
    shower_data_test = load_dl1_data_stereo_list_selected(
        file_list=glob.glob(f_), sub_dict=cfg["energy_rf"], file_n_key="test_file_n"
    )

    # --- Check intersections ---
    wt_ = "WARNING: check only on gammas; use it in classifier to check also protons"
    # useful ONLY if test_file_n == 0
    wn_ = "WARNING: test_file_n != 0, considering only a selection of the test sample"
    if "check_train_test" in cfg["energy_rf"].keys():
        if cfg["energy_rf"]["check_train_test"]:
            info_message("Check train and test", prefix="EnergyRF")
            info_message(wt_, prefix="EnergyRF")
            if cfg["energy_rf"]["test_file_n"] > 0:
                info_message(wn_, prefix="EnergyRF")
            test_passed = check_train_test_intersections(
                shower_data_train, shower_data_test
            )
            s_ = "Test PASSED" if test_passed else "Test NOT PASSED"
            info_message(s_, prefix="EnergyRF")

    # Computing event weights
    info_message("Computing the train sample event weights...", prefix="EnergyRF")
    alt_edges, intensity_edges = compute_event_weights()

    mc_weights = get_weights_mc_dir_class(shower_data_train, alt_edges, intensity_edges)

    shower_data_train = shower_data_train.join(mc_weights)

    tel_ids, tel_ids_LST, tel_ids_MAGIC = check_tel_ids(cfg)

    # --- Data preparation ---
    l_ = ["obs_id", "event_id"]
    shower_data_train["multiplicity"] = (
        shower_data_train["intensity"].groupby(level=l_).count()
    )
    shower_data_test["multiplicity"] = (
        shower_data_test["intensity"].groupby(level=l_).count()
    )

    # Applying the cuts
    shower_data_train = shower_data_train.query(cfg["energy_rf"]["cuts"])
    shower_data_test = shower_data_test.query(cfg["energy_rf"]["cuts"])

    # --- Training the direction RF ---
    info_message("Training the RF\n", prefix="EnergyRF")

    energy_estimator = EnergyEstimatorPandas(
        cfg["energy_rf"]["features"], **cfg["energy_rf"]["settings"]
    )
    energy_estimator.fit(shower_data_train)
    energy_estimator.save(
        os.path.join(cfg["energy_rf"]["save_dir"], cfg["energy_rf"]["joblib_name"])
    )
    # energy_estimator.load(cfg['energy_rf']['save_name'])

    info_message("Parameter importances", prefix="EnergyRF")
    print("")
    r_ = energy_estimator.telescope_regressors
    for tel_id in r_:
        feature_importances = r_[tel_id].feature_importances_
        print(f"  tel_id: {tel_id}")
        z_ = zip(energy_estimator.feature_names, feature_importances)
        for feature, importance in z_:
            print(f"  {feature:.<15s}: {importance:.4f}")
        print("")

    info_message("Applying RF...", prefix="EnergyRF")
    energy_reco = energy_estimator.predict(shower_data_test)
    shower_data_test = shower_data_test.join(energy_reco)

    # Evaluating performance
    info_message("Evaluating performance...", prefix="EnergyRF")

    idx = pd.IndexSlice

    tel_migmatrix = {}
    for tel_id in tel_ids:
        tel_migmatrix[tel_id] = evaluate_performance_energy(
            shower_data_test.loc[idx[:, :, tel_id], ["true_energy", "energy_reco"]],
            "energy_reco",
        )

    migmatrix = evaluate_performance_energy(shower_data_test, "energy_reco_mean")

    # ================
    # === Plotting ===
    # ================
    plt.figure(figsize=tuple(cfg["energy_rf"]["fig_size"]))

    grid_shape = (2, len(tel_ids) + 1)
    # --- PLOT ---
    for index, tel_id in enumerate(tel_ids):
        for i, tel_label in enumerate(cfg["all_tels"]["tel_n"]):
            if tel_id in cfg[tel_label]["tel_ids"]:
                n = cfg["all_tels"]["tel_n_short"][i]
                j = tel_id - cfg[tel_label]["tel_ids"][0] + 1
                name = f"{n}{j}"
        plot_migmatrix(
            index=index, name=name, matrix=tel_migmatrix[tel_id], grid_shape=grid_shape
        )
        index += 1
    # --- GLOBAL ---
    plot_migmatrix(index=index, name="All", matrix=migmatrix, grid_shape=grid_shape)

    plt.tight_layout()
    save_plt(
        n=cfg["energy_rf"]["fig_name"], rdir=cfg["energy_rf"]["save_dir"], vect="pdf",
    )
    plt.close()


if __name__ == "__main__":
    args = PARSER.parse_args()
    kwargs = args.__dict__
    start_time = time.time()
    do_classifier = kwargs["all"] or kwargs["classifier"]
    do_energy = kwargs["all"] or kwargs["energy"]
    do_direction = kwargs["all"] or kwargs["direction"]

    if not (do_classifier or do_energy or do_direction):
        print("No options selected")
        print("Type -h or --help for information on how to run the script")

    if do_classifier:
        train_classifier_rf_stereo(config_file=kwargs["config_file"])
    if do_energy:
        train_energy_rf_stereo(config_file=kwargs["config_file"])
    if do_direction:
        train_direction_rf_stereo(config_file=kwargs["config_file"])

    print_elapsed_time(start_time, time.time())
