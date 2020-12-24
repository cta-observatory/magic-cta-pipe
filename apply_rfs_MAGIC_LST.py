# coding: utf-8

import datetime
import yaml
import time
import argparse
import pandas as pd
import scipy
from astropy import units as u

from magicctapipe.train.event_processing import (
    EnergyEstimatorPandas,
    DirectionEstimatorPandas,
    EventClassifierPandas,
)
from magicctapipe.utils.tels import *
from magicctapipe.utils.utils import *
from magicctapipe.utils.filedir import *

PARSER = argparse.ArgumentParser(
    description="Apply random forests. For stereo data.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
PARSER.add_argument(
    "-cfg",
    "--config_file",
    type=str,
    required=True,
    help="Configuration file, yaml format",
)


def apply_rfs_stereo(config_file):
    """Apply Random Forests

    Parameters
    ----------
    config_file : str
        configuration file
    """
    print_title("Apply RFs")

    # ------------------------------
    # Reading the configuration file
    cfg = load_cfg_file(config_file)

    # Get tel_ids
    tel_ids, tel_ids_LST, tel_ids_MAGIC = check_tel_ids(cfg)

    # --- MAGIC - LST description ---
    array_tel_descriptions = get_array_tel_descriptions(
        tel_ids_LST=tel_ids_LST, tel_ids_MAGIC=tel_ids_MAGIC
    )

    # Using only the "mc" and "data" "test_sample"
    data_types = ["mc", "data"]
    sample = "test_sample"

    for data_type in data_types:
        info_message(f'Loading "{data_type}", sample "{sample}"', prefix="ApplyRF")

        file_list = glob.glob(cfg["data_files"][data_type][sample]["hillas_h5"])

        for file in file_list:
            print(f"Analyzing file:\n{file}")
            out_file = os.path.join(
                os.path.dirname(cfg["data_files"][data_type][sample]["reco_h5"]),
                out_file_h5_reco(in_file=file),
            )
            print(f"Output file:\n{out_file}")

            shower_data = load_dl1_data_stereo(file)

            # Dropping data with the wrong altitude
            shower_data = shower_data.query(cfg["global"]["wrong_alt"])

            # Computing the event "multiplicity"
            l_ = ["obs_id", "event_id"]
            shower_data["multiplicity"] = (
                shower_data["intensity"].groupby(level=l_).count()
            )

            # Added by Lea Heckmann 2020-05-15 for the moment to delete duplicate
            # events
            info_message(f"Removing duplicate events", prefix="ApplyRF")
            shower_data = shower_data[~shower_data.index.duplicated()]

            # Applying RFs of every kind
            rf_kinds = ["direction_rf", "energy_rf", "classifier_rf"]
            for rf_kind in rf_kinds:
                info_message(f"Loading RF: {rf_kind}", prefix="ApplyRF")

                if rf_kind == "direction_rf":
                    estimator = DirectionEstimatorPandas(
                        cfg[rf_kind]["features"],
                        array_tel_descriptions,
                        **cfg[rf_kind]["settings"],
                    )
                elif rf_kind == "energy_rf":
                    estimator = EnergyEstimatorPandas(
                        cfg[rf_kind]["features"], **cfg[rf_kind]["settings"]
                    )

                elif rf_kind == "classifier_rf":
                    estimator = EventClassifierPandas(
                        cfg[rf_kind]["features"], **cfg[rf_kind]["settings"]
                    )

                estimator.load(
                    os.path.join(cfg[rf_kind]["save_dir"], cfg[rf_kind]["joblib_name"])
                )

                # --- Applying RF ---
                info_message(f"Applying RF: {rf_kind}", prefix="ApplyRF")
                reco = estimator.predict(shower_data)

                # Appeding the result to the main data frame
                shower_data = shower_data.join(reco)
            # --- END LOOP on rf_kinds ---

            # --- Store ---
            info_message("Saving the reconstructed data", prefix="ApplyRF")
            # Storing the reconstructed values for the given data sample
            shower_data.to_hdf(out_file, key="dl2/reco")
            # Take mc_header form dl1 and save in dl2
            mc_ = pd.read_hdf(file, key="dl1/mc_header")
            mc_.to_hdf(out_file, key="dl2/mc_header")
        # --- END LOOP on file_list ---

    # --- END LOOP on data_types ---


if __name__ == "__main__":
    args = PARSER.parse_args()
    kwargs = args.__dict__
    start_time = time.time()
    apply_rfs_stereo(config_file=kwargs["config_file"],)
    print_elapsed_time(start_time, time.time())
