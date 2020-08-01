# coding: utf-8

import argparse
import yaml
import datetime

import pandas as pd
import numpy as np

import sklearn.metrics

from ctapipe.reco.event_processing import EventClassifierPandas

from astropy import units as u

from matplotlib import pyplot, colors


def info_message(text, prefix='info'):
    """
    This function prints the specified text with the prefix of the current date

    Parameters
    ----------
    text: str

    Returns
    -------
    None

    """

    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"({prefix:s}) {date_str:s}: {text:s}")


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
        class_names = list(filter(lambda name: 'event_class_' in name and '_mean' in name, data.columns))
    else:
        class_names = list(filter(lambda name: 'event_class_' in name and '_mean' not in name, data.columns))
        
    proba = data[class_names].values
    predicted_class = proba.argmax(axis=1)

    report['metrics']['acc'] = sklearn.metrics.accuracy_score(data['true_event_class'], predicted_class)
    
    true_class = np.clip(data['true_event_class'], 0, 1)
    true_class = 1 - true_class
    report['metrics']['auc_roc'] = sklearn.metrics.roc_auc_score(true_class, proba[:, 0])

    return report


def load_data_sample(sample):
    shower_data = pd.DataFrame()

    for telescope in sample:
        info_message(f'Loading {telescope} data...', prefix='ClassifierRF')
        
        hillas_data = pd.read_hdf(sample[telescope]['hillas_output'], key='dl1/hillas_params')
        hillas_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
        
        shower_data = shower_data.append(hillas_data)
        
    shower_data.sort_index(inplace=True)
    
    return shower_data


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
    mc_intensity_bins[mc_intensity_bins == len(intensity_edges) - 1] = len(intensity_edges) - 2

    mc_weights = 1 / mc_hist[mc_alt_bins, mc_intensity_bins]
    mc_weights *= availability_hist[mc_alt_bins, mc_intensity_bins]
    
    # --- Bkg weights ---
    bkg_alt_bins = np.digitize(bkg_data['tel_alt'], alt_edges) - 1
    bkg_intensity_bins = np.digitize(bkg_data['intensity'], intensity_edges) - 1

    # Treating the out-of-range events
    bkg_alt_bins[bkg_alt_bins == len(alt_edges) - 1] = len(alt_edges) - 2
    bkg_intensity_bins[bkg_intensity_bins == len(intensity_edges) - 1] = len(intensity_edges) - 2

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

# --------------------------
# Adding the argument parser
arg_parser = argparse.ArgumentParser(description="""
This tools fits the event classification random forest on the specified events files.
""")

arg_parser.add_argument("--config", default="config.yaml",
                        help='Configuration file to steer the code execution.')

parsed_args = arg_parser.parse_args()
# --------------------------

# ------------------------------
# Reading the configuration file

file_not_found_message = """
Error: can not load the configuration file {:s}.
Please check that the file exists and is of YAML or JSON format.
Exiting.
"""

try:
    config = yaml.safe_load(open(parsed_args.config, "r"))
except IOError:
    print(file_not_found_message.format(parsed_args.config))
    exit()

if 'classifier_rf' not in config:
    print('Error: the configuration file is missing the "classifier_rf" section. Exiting.')
    exit()
# ------------------------------


# --------------------
# --- Train sample ---
info_message('Loading MC train data...', prefix='ClassifierRF')
mc_data = load_data_sample(config['data_files']['mc']['train_sample'])

info_message('Loading "off" train data...', prefix='ClassifierRF')
bkg_data = load_data_sample(config['data_files']['data']['train_sample'])

# True event classes
mc_data['true_event_class'] = 0
bkg_data['true_event_class'] = 1

# Dropping data with the wrong altitude
bkg_data = bkg_data.query('tel_alt < 1.5707963267948966')

# Dropping extra keys
bkg_data.drop('mjd', axis=1, inplace=True)
mc_data.drop(['true_energy', 'true_alt', 'true_az'], axis=1, inplace=True)

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
#print(shower_data_train.head())
# --------------------

# -------------------
# --- Test sample ---
info_message('Loading MC test data...', prefix='ClassifierRF')
mc_data = load_data_sample(config['data_files']['mc']['test_sample'])

info_message('Loading "off" test data...', prefix='ClassifierRF')
bkg_data = load_data_sample(config['data_files']['data']['test_sample'])

# True event classes
mc_data['true_event_class'] = 0
bkg_data['true_event_class'] = 1

# Dropping data with the wrong altitude
bkg_data = bkg_data.query('tel_alt < 1.5707963267948966')

# Dropping extra keys
bkg_data.drop('mjd', axis=1, inplace=True)
mc_data.drop(['true_energy', 'true_alt', 'true_az'], axis=1, inplace=True)

# Merging the test sample
shower_data_test = mc_data.append(bkg_data)
# -------------------


info_message('Preprosessing...', prefix='ClassifierRF')

# --- Data preparation ---
shower_data_train['multiplicity'] = shower_data_train['intensity'].groupby(level=['obs_id', 'event_id']).count()
shower_data_test['multiplicity'] = shower_data_test['intensity'].groupby(level=['obs_id', 'event_id']).count()

# Applying the cuts
shower_data_train = shower_data_train.query(config['classifier_rf']['cuts'])
shower_data_test = shower_data_test.query(config['classifier_rf']['cuts'])

# --- Training the direction RF ---
info_message('Training RF...', prefix='ClassifierRF')

class_estimator = EventClassifierPandas(config['classifier_rf']['features'], 
                                         **config['classifier_rf']['settings'])
class_estimator.fit(shower_data_train)
class_estimator.save(config['classifier_rf']['save_name'])
#class_estimator.load(config['classifier_rf']['save_name'])

info_message('Parameter importances', prefix='ClassifierRF')
print('')
for tel_id in class_estimator.telescope_classifiers:
    feature_importances = class_estimator.telescope_classifiers[tel_id].feature_importances_

    print(f'  tel_id: {tel_id}')
    for feature, importance in zip(class_estimator.feature_names, feature_importances):
        print(f"  {feature:.<15s}: {importance:.4f}")
    print('')

info_message('Applying RF...', prefix='ClassifierRF')
class_reco = class_estimator.predict(shower_data_test)
shower_data_test = shower_data_test.join(class_reco)

# Evaluating performance
info_message('Evaluating performance...', prefix='ClassifierRF')

idx = pd.IndexSlice

performance = dict()
tel_ids = shower_data_test.index.levels[2]

performance[0] = evaluate_performance(shower_data_test.loc[idx[:, :, 1], shower_data_test.columns],
                                      class0_name='event_class_0_mean')

for tel_id in tel_ids:
    performance[tel_id] = evaluate_performance(shower_data_test.loc[idx[:, :, tel_id], shower_data_test.columns])

# ================
# === Plotting ===
# ================

#pyplot.style.use('presentation')

pyplot.figure(figsize=(20, 10))

grid_shape = (2, 3)

for tel_num, tel_id in enumerate(performance):
    pyplot.subplot2grid(grid_shape, (0, tel_num))
    pyplot.title(f'Tel {tel_id} estimation')
    pyplot.xlabel('Class 0 probability')
    pyplot.ylabel('Event density')

    gammaness = performance[tel_id]['gammaness']
    print(performance[tel_id]['metrics'])

    for class_i, event_class in enumerate(gammaness):
        pyplot.step(gammaness[event_class]['XEdges'][:-1], 
                    gammaness[event_class]['Hist'],
                    where='post',
                    color=f'C{class_i}',
                    label=f'Class {event_class}')
        
        pyplot.step(gammaness[event_class]['XEdges'][1:], 
                    gammaness[event_class]['Hist'],
                    where='pre',
                    color=f'C{class_i}')
    
        pyplot.fill_between(gammaness[event_class]['XEdges'][:-1], 
                            gammaness[event_class]['Hist'],
                            step='post',
                            color=f'C{class_i}',
                            alpha=0.3)
        
        value = performance[tel_id]['metrics']['acc']
        pyplot.text(0.9, 0.9, f"acc={value:.2f}",
                    ha='right', va='top',
                    transform=pyplot.gca().transAxes
                    )
        
        value = performance[tel_id]['metrics']['auc_roc']
        pyplot.text(0.9, 0.8, f"auc_roc={value:.2f}",
                    ha='right', va='top',
                    transform=pyplot.gca().transAxes
                    )
    
    pyplot.legend()
    
for tel_num, tel_id in enumerate(performance):
    pyplot.subplot2grid(grid_shape, (1, tel_num))
    pyplot.semilogy()
    pyplot.title(f'Tel {tel_id} estimation')
    pyplot.xlabel('Class 0 probability')
    pyplot.ylabel('Cumulative probability')
    pyplot.ylim(1e-3, 1)

    gammaness = performance[tel_id]['gammaness']

    for class_i, event_class in enumerate(gammaness):
        pyplot.step(gammaness[event_class]['XEdges'][:-1], 
                    gammaness[event_class]['Cumsum'],
                    where='post',
                    color=f'C{class_i}',
                    label=f'Class {event_class}')
        
        pyplot.step(gammaness[event_class]['XEdges'][1:], 
                    gammaness[event_class]['Cumsum'],
                    where='pre',
                    color=f'C{class_i}')
    
        pyplot.fill_between(gammaness[event_class]['XEdges'][:-1], 
                            gammaness[event_class]['Cumsum'],
                            step='post',
                            color=f'C{class_i}',
                            alpha=0.3)
    
    pyplot.legend()

pyplot.tight_layout()

pyplot.savefig('classifier_rf_gammaness.png')
pyplot.close()
