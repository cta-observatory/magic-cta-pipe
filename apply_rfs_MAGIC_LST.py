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
    EventClassifierPandas
)
from magicctapipe.utils.tels import *
from magicctapipe.utils.utils import *
from magicctapipe.utils.filedir import *

PARSER = argparse.ArgumentParser(
    description=("This tools fits the event classification random forest on "
                 "the specified events files. For stereo data."),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
PARSER.add_argument('-cfg', '--config_file', type=str, required=True,
                    help='Configuration file to steer the code execution')


# =================
# === Main code ===
# =================
def apply_rfs_stereo(config_file):
    # ------------------------------
    # Reading the configuration file
    cfg = load_cfg_file(config_file)

    # Using only the "data" "test_sample"
    data_types = ['mc', 'data']
    sample = 'test_sample'

    for data_type in data_types:
        info_message(f'Loading "{data_type}", sample "{sample}"',
                     prefix='ApplyRF')

        shower_data = load_dl1_data_stereo(
            file=cfg['data_files'][data_type][sample]['hillas_h5'])

        # Dropping data with the wrong altitude
        shower_data = shower_data.query(cfg['global']['wrong_alt'])

        # Computing the event "multiplicity"
        l_ = ['obs_id', 'event_id']
        shower_data['multiplicity'] = \
            shower_data['intensity'].groupby(level=l_).count()

        # Added by Lea Heckmann 2020-05-15 for the moment to delete duplicate
        # events
        info_message(f'Removing duplicate events', prefix='ApplyRF')
        shower_data = shower_data[~shower_data.index.duplicated()]

        # Get tel_ids
        tel_ids, tel_ids_LST, tel_ids_MAGIC = \
            intersec_tel_ids(
                tel_ids_sel=get_tel_ids_dl1(shower_data),
                all_tel_ids_LST=cfg['LST']['tel_ids'],
                all_tel_ids_MAGIC=cfg['MAGIC']['tel_ids']
            )

        # --- MAGIC - LST description ---
        array_tel_descriptions = get_array_tel_descriptions(
            tel_ids_LST=tel_ids_LST,
            tel_ids_MAGIC=tel_ids_MAGIC
        )

        # Applying RFs of every kind
        for rf_kind in ['direction_rf', 'energy_rf', 'classifier_rf']:
            info_message(f'Loading RF: {rf_kind}', prefix='ApplyRF')

            if rf_kind == 'direction_rf':
                estimator = DirectionEstimatorPandas(
                    cfg[rf_kind]['features'],
                    array_tel_descriptions,
                    **cfg[rf_kind]['settings']
                )
            elif rf_kind == 'energy_rf':
                estimator = EnergyEstimatorPandas(
                    cfg[rf_kind]['features'],
                    **cfg[rf_kind]['settings']
                )

            elif rf_kind == 'classifier_rf':
                estimator = EventClassifierPandas(
                    cfg[rf_kind]['features'],
                    **cfg[rf_kind]['settings']
                )

            estimator.load(os.path.join(cfg[rf_kind]['save_dir'],
                                        cfg[rf_kind]['joblib_name']))

            # --- Applying RF ---
            info_message(f'Applying RF: {rf_kind}', prefix='ApplyRF')
            reco = estimator.predict(shower_data)

            # Appeding the result to the main data frame
            shower_data = shower_data.join(reco)

        # --- Store ---
        info_message('Saving the reconstructed data', prefix='ApplyRF')
        # Storing the reconstructed values for the given data sample
        shower_data.to_hdf(cfg['data_files'][data_type][sample]['reco_h5'],
                           key='dl2/reco')
        # Take mc_header form dl1 and save in dl2
        mc_ = pd.read_hdf(cfg['data_files'][data_type][sample]['hillas_h5'],
                          key='dl1/mc_header')
        mc_.to_hdf(cfg['data_files'][data_type][sample]['reco_h5'],
                   key='dl2/mc_header')


if __name__ == '__main__':
    args = PARSER.parse_args()
    kwargs = args.__dict__
    start_time = time.time()
    apply_rfs_stereo(
        config_file=kwargs['config_file'],
    )
    print("Execution time: %.2f s" % (time.time() - start_time))
