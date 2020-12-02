# coding: utf-8

import argparse
import time
import pandas as pd
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

from magicctapipe.utils.plot import *
from magicctapipe.utils.utils import *
from magicctapipe.utils.tels import *
from magicctapipe.utils.filedir import *
from magicctapipe.train.event_processing import EventClassifierPandas

PARSER = argparse.ArgumentParser(
    description=("This tools fits the event classification random forest on "
                 "the specified events files. For stereo data."),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
PARSER.add_argument('-cfg', '--config_file', type=str, required=True,
                    help='Configuration file to steer the code execution')


def GetHist(data, bins=30, range=None, weights=None):
    hs, edges = np.histogram(data, bins=bins, range=range, weights=weights)
    loc = (edges[1:] + edges[:-1]) / 2

    hist = {}
    hist['Hist'] = hs
    hist['X'] = loc
    hist['XEdges'] = edges

    return hist


def evaluate_performance(data, class0_name='event_class_0'):
    data = data.dropna()

    report = {
        "gammaness": dict(),
        "metrics": dict()
    }

    for event_class in data['true_event_class'].unique():
        events = data.query(f'true_event_class == {event_class}')
        hist = GetHist(events[class0_name], bins=100, range=(0, 1))
        hist['Hist'] = hist['Hist'] / hist['Hist'].sum()
        hist['Cumsum'] = 1 - np.cumsum(hist['Hist'])

        report['gammaness'][event_class] = hist

    if 'mean' in class0_name:
        class_names = list(filter(lambda name: 'event_class_' in name
                                  and '_mean' in name, data.columns))
    else:
        class_names = list(filter(lambda name: 'event_class_' in name
                                  and '_mean' not in name, data.columns))

    proba = data[class_names].values
    predicted_class = proba.argmax(axis=1)

    report['metrics']['acc'] = sklearn.metrics.accuracy_score(
        data['true_event_class'], predicted_class
    )

    true_class = np.clip(data['true_event_class'], 0, 1)
    true_class = 1 - true_class

    report['metrics']['auc_roc'] = sklearn.metrics.roc_auc_score(
        true_class, proba[:, 0]
    )

    return report


def get_weights(mc_data, bkg_data, alt_edges, intensity_edges):
    mc_hist, _, _ = np.histogram2d(mc_data['tel_alt'],
                                   mc_data['intensity'],
                                   bins=[alt_edges, intensity_edges])
    bkg_hist, _, _ = np.histogram2d(bkg_data['tel_alt'],
                                    bkg_data['intensity'],
                                    bins=[alt_edges, intensity_edges])

    availability_hist = np.clip(mc_hist, 0, 1) * np.clip(bkg_hist, 0, 1)

    # --- MC weights ---
    mc_alt_bins = np.digitize(mc_data['tel_alt'], alt_edges) - 1
    mc_intensity_bins = np.digitize(mc_data['intensity'], intensity_edges) - 1

    # Treating the out-of-range events
    mc_alt_bins[mc_alt_bins == len(alt_edges) - 1] = len(alt_edges) - 2
    mc_intensity_bins[mc_intensity_bins == len(intensity_edges) - 1] = \
        len(intensity_edges) - 2

    mc_weights = 1 / mc_hist[mc_alt_bins, mc_intensity_bins]
    mc_weights *= availability_hist[mc_alt_bins, mc_intensity_bins]

    # --- Bkg weights ---
    bkg_alt_bins = np.digitize(bkg_data['tel_alt'], alt_edges) - 1
    bkg_intensity_bins = np.digitize(
        bkg_data['intensity'], intensity_edges) - 1

    # Treating the out-of-range events
    bkg_alt_bins[bkg_alt_bins == len(alt_edges) - 1] = len(alt_edges) - 2
    bkg_intensity_bins[bkg_intensity_bins == len(intensity_edges) - 1] = \
        len(intensity_edges) - 2

    bkg_weights = 1 / bkg_hist[bkg_alt_bins, bkg_intensity_bins]
    bkg_weights *= availability_hist[bkg_alt_bins, bkg_intensity_bins]

    # --- Storing to a data frame ---
    mc_weight_df = pd.DataFrame(data={'event_weight': mc_weights},
                                index=mc_data.index)
    bkg_weight_df = pd.DataFrame(data={'event_weight': bkg_weights},
                                 index=bkg_data.index)

    return mc_weight_df, bkg_weight_df


# =================
# === Main code ===
# =================
def train_classifier_rf_stereo(config_file):
    # --- Reading the configuration file ---
    cfg = load_cfg_file_check(config_file=config_file, label='classifier_rf')

    # --- Check output directory ---
    check_folder(cfg['classifier_rf']['save_dir'])

    # --- Train sample ---
    f_ = cfg['data_files']['mc']['train_sample']['hillas_h5']
    info_message('Loading MC train data...', prefix='ClassifierRF')
    mc_data = load_dl1_data(f_)

    f_ = cfg['data_files']['data']['train_sample']['hillas_h5']
    info_message('Loading "off" train data...', prefix='ClassifierRF')
    bkg_data = load_dl1_data(f_)

    # True event classes
    mc_data['true_event_class'] = 0
    bkg_data['true_event_class'] = 1

    # Dropping data with the wrong altitude
    bkg_data = bkg_data.query(cfg['global']['wrong_alt'])

    # Dropping extra keys
    # bkg_data.drop('mjd', axis=1, inplace=True) # Key doesn't exist in data
    mc_data.drop(cfg['classifier_rf']['extra_keys'], axis=1, inplace=True)

    # Computing event weights
    sin_edges = np.linspace(0, 1, num=51)
    alt_edges = np.lib.scimath.arcsin(sin_edges)
    intensity_edges = np.logspace(1, 5, num=51)

    mc_weights, bkg_weights = get_weights(mc_data, bkg_data,
                                          alt_edges, intensity_edges)

    mc_data = mc_data.join(mc_weights)
    bkg_data = bkg_data.join(bkg_weights)

    # Merging the train sample
    shower_data_train = mc_data.append(bkg_data)
    # --------------------

    # --- Test sample ---
    f_ = cfg['data_files']['mc']['test_sample']['hillas_h5']
    info_message('Loading MC test data...', prefix='ClassifierRF')
    mc_data = load_dl1_data(f_)

    f_ = cfg['data_files']['data']['test_sample']['hillas_h5']
    info_message('Loading "off" test data...', prefix='ClassifierRF')
    bkg_data = load_dl1_data(f_)

    # True event classes
    mc_data['true_event_class'] = 0
    bkg_data['true_event_class'] = 1

    # Dropping data with the wrong altitude
    bkg_data = bkg_data.query(cfg['global']['wrong_alt'])

    # Dropping extra keys
    # bkg_data.drop('mjd', axis=1, inplace=True) # Key doesn't exist in data
    mc_data.drop(cfg['classifier_rf']['extra_keys'], axis=1, inplace=True)
    # !!! CHECK !!!
    bkg_data.drop(cfg['classifier_rf']['extra_keys'], axis=1, inplace=True)

    # Merging the test sample
    shower_data_test = mc_data.append(bkg_data)
    # -------------------

    info_message('Preprosessing...', prefix='ClassifierRF')

    # --- Data preparation ---
    l_ = ['obs_id', 'event_id']
    shower_data_train['multiplicity'] = \
        shower_data_train['intensity'].groupby(level=l_).count()
    shower_data_test['multiplicity'] = \
        shower_data_test['intensity'].groupby(level=l_).count()

    # Applying the cuts
    shower_data_train = shower_data_train.query(cfg['classifier_rf']['cuts'])
    shower_data_test = shower_data_test.query(cfg['classifier_rf']['cuts'])

    # --- Training the direction RF ---
    info_message('Training RF...', prefix='ClassifierRF')

    class_estimator = EventClassifierPandas(cfg['classifier_rf']['features'],
                                            **cfg['classifier_rf']['settings'])
    class_estimator.fit(shower_data_train)
    class_estimator.save(os.path.join(cfg['classifier_rf']['save_dir'],
                                      cfg['classifier_rf']['joblib_name']))
    # class_estimator.load(cfg['classifier_rf']['save_name'])

    info_message('Parameter importances', prefix='ClassifierRF')
    print('')
    for tel_id in class_estimator.telescope_classifiers:
        feature_importances = \
            class_estimator.telescope_classifiers[tel_id].feature_importances_

        print(f'  tel_id: {tel_id}')
        z_ = zip(class_estimator.feature_names, feature_importances)
        for feature, importance in z_:
            print(f"  {feature:.<15s}: {importance:.4f}")
        print('')

    info_message('Applying RF...', prefix='ClassifierRF')
    class_reco = class_estimator.predict(shower_data_test)
    shower_data_test = shower_data_test.join(class_reco)

    # Evaluating performance
    info_message('Evaluating performance...', prefix='ClassifierRF')

    idx = pd.IndexSlice

    performance = dict()
    # tel_ids = shower_data_test.index.levels[2]

    tel_ids, tel_ids_LST, tel_ids_MAGIC = \
        intersec_tel_ids(
            tel_ids_sel=get_tel_ids_dl1(shower_data_test),
            all_tel_ids_LST=cfg['LST']['tel_ids'],
            all_tel_ids_MAGIC=cfg['MAGIC']['tel_ids']
        )
    # !!! CHECK !!!
    performance[0] = evaluate_performance(
        shower_data_test.loc[idx[:, :, tel_ids[0]], shower_data_test.columns],
        class0_name='event_class_0_mean'
    )

    for tel_id in tel_ids:
        performance[tel_id] = evaluate_performance(
            shower_data_test.loc[idx[:, :, tel_id], shower_data_test.columns])

    # ================
    # === Plotting ===
    # ================

    plt.figure(figsize=tuple(cfg['classifier_rf']['fig_size']))
    labels = ['Gammaness', 'Hadroness']

    grid_shape = (2, len(tel_ids)+1)

    for tel_num, tel_id in enumerate(performance):
        plt.subplot2grid(grid_shape, (0, tel_num))
        if(tel_id == 0):
            plt.title(f'Tel {tel_id} estimation')
        else:
            n_ = get_tel_name(tel_id=tel_id, cfg=cfg)
            plt.title(f'{n_} estimation')
        plt.xlabel('Gamma probability')
        # plt.xlabel('Class 0 probability')
        plt.ylabel('Event density')

        gammaness = performance[tel_id]['gammaness']
        print(performance[tel_id]['metrics'])

        for class_i, event_class in enumerate(gammaness):
            plt.step(gammaness[event_class]['XEdges'][:-1],
                     gammaness[event_class]['Hist'],
                     where='post',
                     color=f'C{class_i}',
                     label=labels[event_class])
            #  label=f'Class {event_class}')

            plt.step(gammaness[event_class]['XEdges'][1:],
                     gammaness[event_class]['Hist'],
                     where='pre',
                     color=f'C{class_i}')

            plt.fill_between(gammaness[event_class]['XEdges'][:-1],
                             gammaness[event_class]['Hist'],
                             step='post',
                             color=f'C{class_i}',
                             alpha=0.3)

            value = performance[tel_id]['metrics']['acc']
            plt.text(0.9, 0.9, f"acc={value:.2f}",
                     ha='right', va='top',
                     transform=plt.gca().transAxes
                     )

            value = performance[tel_id]['metrics']['auc_roc']
            plt.text(0.9, 0.8, f"auc_roc={value:.2f}",
                     ha='right', va='top',
                     transform=plt.gca().transAxes
                     )

        plt.legend()

    for tel_num, tel_id in enumerate(performance):
        plt.subplot2grid(grid_shape, (1, tel_num))
        plt.semilogy()
        if(tel_id == 0):
            plt.title(f'Tel {tel_id} estimation')
        else:
            n_ = get_tel_name(tel_id=tel_id, cfg=cfg)
            plt.title(f'{n_} estimation')
        plt.xlabel('Gamma probability')
        # plt.xlabel('Class 0 probability')
        plt.ylabel('Cumulative probability')
        plt.ylim(1e-3, 1)

        gammaness = performance[tel_id]['gammaness']

        for class_i, event_class in enumerate(gammaness):
            plt.step(gammaness[event_class]['XEdges'][:-1],
                     gammaness[event_class]['Cumsum'],
                     where='post',
                     color=f'C{class_i}',
                     label=labels[event_class])
            #  label=f'Class {event_class}')

            plt.step(gammaness[event_class]['XEdges'][1:],
                     gammaness[event_class]['Cumsum'],
                     where='pre',
                     color=f'C{class_i}')

            plt.fill_between(gammaness[event_class]['XEdges'][:-1],
                             gammaness[event_class]['Cumsum'],
                             step='post',
                             color=f'C{class_i}',
                             alpha=0.3)

        plt.legend()

    plt.tight_layout()
    save_plt(n=cfg['classifier_rf']['fig_name'],
             rdir=cfg['classifier_rf']['save_dir'],
             vect='')

    plt.close()


if __name__ == '__main__':
    args = PARSER.parse_args()
    kwargs = args.__dict__
    start_time = time.time()
    train_classifier_rf_stereo(
        config_file=kwargs['config_file'],
    )
    print("Execution time: %.2f s" % (time.time() - start_time))
