import os
import sys
import argparse
import yaml
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import astropy
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from regions import CircleSkyRegion

import gammapy
from gammapy.maps import Map, MapAxis
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
    SpectrumDatasetOnOff,
    FluxPointsDataset,
)
from gammapy.modeling.models import (
    LogParabolaSpectralModel,
    PowerLawSpectralModel,
    create_crab_spectral_model,
    SkyModel,
)
from gammapy.makers import (
    SafeMaskMaker,
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
)
from gammapy.estimators import FluxPointsEstimator
from gammapy.visualization import plot_spectrum_datasets_off_regions
from gammapy.makers.utils import make_theta_squared_table
from gammapy.visualization import plot_theta_squared_table

def crab_magic_new(e):
    """
    Magic "new" (arXiv:1406.6892) Crab nebula spectrum.
    Parameters
    ----------
    e: array_like
        Energy in eV.
    Returns
    -------
    array_like:
        Crab E^2xdN/dE spectrum.
    """

    e0 = 1 * u.TeV
    norm = 3.23e-23 / (u.eV * u.cm**2 * u.s)
    
    dnde = norm * (e.to(u.eV)/e0)**(-2.47 - 0.24*np.log10(e/e0))
    e2dnde = dnde * e.to(u.eV)**2

    return e2dnde

def crab_magic_2(e):
    """
    Magic "new" (arXiv:1409.5594) Crab nebula spectrum.

    Parameters
    ----------
    e: array_like
        Energy in eV.

    Returns
    -------
    array_like:
        Crab E^2xdN/dE spectrum.
    """

    e0 = 1 * u.TeV
    norm = 3.395e-23 / (u.eV * u.cm**2 * u.s)
    
    dnde = norm * (e.to(u.eV)/e0)**(-2.511 - 0.2143*np.log10(e/e0))
    e2dnde = dnde * e.to(u.eV)**2

    return e2dnde

def read_config(config):
    options = {}
    options["event_files"] = config["spectrum"]["input_files"]
    options["irf_file"] = config["irf"]["output_name"]
    options["source_name"] = config["source"]["name"]
    options["source_ra"] = config["source"]["coordinates"]["ra_dec"][0]
    options["source_dec"] = config["source"]["coordinates"]["ra_dec"][1]
    options["off_positions"] = config["spectrum"]["off_positions"]
    options["energy_reco_nbins"] = config["spectrum"]["energy_reco_nbins"]
    options["energy_reco_min"] = config["spectrum"]["energy_reco_min"]
    options["energy_reco_max"] = config["spectrum"]["energy_reco_max"]
    options["energy_true_factor"] = config["spectrum"]["energy_true_factor"]
    options["spectral_model"] = config["spectrum"]["spectral_model"]

    return options

print(f"Using numpy version {np.__version__}")
print(f"Using astropy version {astropy.__version__}")
print(f"Using gammapy version {gammapy.__version__}")

arg_parser = argparse.ArgumentParser(description="""
This tools computes the Hillas parameters for the specified data sets.
""")

arg_parser.add_argument("--config", default="config.yaml",
                        help='Configuration file to steer the code execution.')

parsed_args = arg_parser.parse_args()

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

# copying IRF file into proper directory, otherwise it will not be found

options = read_config(config)

event_files = options["event_files"]
irf_file    = Path(options["irf_file"])
event_path  = str(Path(event_files).parent)
os.environ['CALDB'] = event_path
source_name = options["source_name"]
irf_path = Path(f"{event_path}/data/magic/dev/bcf/{source_name}")
irf_path.mkdir(parents=True, exist_ok=True)
shutil.copy(irf_file, irf_path)

event_files_list = sorted([str(filename) for filename in Path(event_files).parent.expanduser().glob(Path(event_files).name)])
data_store = DataStore.from_events_files(event_files_list)
data_store.info()
print(data_store.hdu_table)
print(data_store.obs_table[:][["OBS_ID", "DATE-OBS", "RA_PNT", "DEC_PNT", "OBJECT"]])

obs = data_store.obs(data_store.obs_table[0]["OBS_ID"])
obs.aeff.to_effective_area_table(Angle('0.4d')).plot()
obs.events.select_offset([0, 2.5] * u.deg).peek()
obs.aeff.peek()
obs.edisp.peek()
obs.psf.peek()

source_ra  = options["source_ra"]
source_dec = options["source_dec"]
target_position = SkyCoord(ra=source_ra, dec=source_dec, unit="deg", frame="icrs")
on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)
theta2_axis = MapAxis.from_bounds(0, 0.2, nbin=20, interp="lin", unit="deg2")

obs_ids = list(data_store.obs_table[:]["OBS_ID"])
observations = data_store.get_observations(obs_ids)
theta2_table = make_theta_squared_table(
    observations=observations,
    position=target_position,
    theta_squared_axis=theta2_axis,
)

plt.figure(figsize=(10, 5))
plot_theta_squared_table(theta2_table)
plt.show()

e_reco_nbins  = options["energy_reco_nbins"]
e_reco_min    = options["energy_reco_min"]
e_reco_max    = options["energy_reco_max"]
e_true_factor = options["energy_true_factor"]
e_true_nbins  = int(float(e_reco_nbins)/e_true_factor)
print(e_true_nbins)

e_reco = MapAxis.from_energy_bounds(e_reco_min, e_reco_max, e_reco_nbins, unit="TeV", name="energy")
print(e_reco.edges)
e_true = MapAxis.from_energy_bounds(e_reco_min, e_reco_max, e_true_nbins, unit="TeV", name="energy_true")
print(e_true.edges)

dataset_empty = SpectrumDataset.create(
    e_reco=e_reco, e_true=e_true, region=on_region
)
dataset_maker = SpectrumDatasetMaker(
    containment_correction=False, selection=["counts", "exposure", "edisp"]
)

exclusion_regions = []

if "exclusion_region" in config["spectrum"]:
    for excluded_source in config["spectrum"]["exclusion_region"].values():
        ra     = excluded_source[0]
        dec    = excluded_source[1]
        radius = excluded_source[2]
        exclusion_region = CircleSkyRegion(
                            center=SkyCoord(ra, dec, unit="deg", frame="galactic"),
                            radius=radius * u.deg,
                            )
        exclusion_regions.append(exclusion_region)

skydir = target_position.galactic
exclusion_mask = Map.create(
    npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs"
)
mask = exclusion_mask.geom.region_mask(exclusion_regions, inside=False)
exclusion_mask.data = mask
exclusion_mask.plot()
off_positions = options["off_positions"]
bkg_maker = ReflectedRegionsBackgroundMaker(max_region_number=off_positions)

safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

datasets = Datasets()

for obs_id, observation in zip(obs_ids, observations):
    dataset = dataset_maker.run(
        dataset_empty.copy(name=str(obs_id)), observation
    )
    print(observation.pointing_radec)
    print(observation.pointing_radec.separation(on_region.center).deg)
    dataset_on_off = bkg_maker.run(dataset, observation)
    dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
    datasets.append(dataset_on_off)

plt.figure(figsize=(8, 8))
_, ax, _ = exclusion_mask.plot()
on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="k")
plot_spectrum_datasets_off_regions(ax=ax, datasets=datasets)
plt.show()

datasets[2].peek()

info_table = datasets.info_table(cumulative=True)
info_table

plt.plot(
    info_table["livetime"].to("h"), info_table["excess"], marker="o", ls="none"
)
plt.xlabel("Livetime [h]")
plt.ylabel("Excess")
plt.show()

plt.plot(
    info_table["livetime"].to("h"),
    info_table["sqrt_ts"],
    marker="o",
    ls="none",
)
plt.xlabel("Livetime [h]")
plt.ylabel("Sqrt(TS)")
plt.show()

if "power_law" in options["spectral_model"]:
    parameters = options["spectral_model"]["power_law"]
    index = parameters["index"]
    amplitude = parameters["amplitude"]
    reference = parameters["reference"]
    print(parameters["index"])
    print(parameters["amplitude"])
    print(parameters["reference"])
    spectral_model = PowerLawSpectralModel(
        index=index, amplitude=amplitude * u.Unit("cm-2 s-1 TeV-1"), reference=reference * u.TeV
    )
elif "log_parabola" in options["spectral_model"]:
    parameters = options["spectral_model"]["log_parabola"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    amplitude = parameters["amplitude"]
    reference = parameters["reference"]
    spectral_model = LogParabolaSpectralModel(
        alpha=alpha, amplitude=amplitude * u.Unit("cm-2 s-1 TeV-1"),beta=beta, reference=reference * u.TeV
    )

model = SkyModel(spectral_model=spectral_model, name="crab")

for dataset in datasets:
    dataset.models = model

fit_joint = Fit(datasets)
result_joint = fit_joint.run()

# we make a copy here to compare it later
model_best_joint = model.copy()
print(result_joint)

ax_spectrum, ax_residuals = datasets[0].plot_fit()
ax_spectrum.set_ylim(0.1, 40)

e_min, e_max = 0.1, 10
energy_edges = np.logspace(np.log10(e_min), np.log10(e_max), 11) * u.TeV
print(energy_edges)

fpe = FluxPointsEstimator(energy_edges=energy_edges, source="crab")
flux_points = fpe.run(datasets=datasets)
print(flux_points.table_formatted)

crab_energy = np.geomspace(0.1, 10, num=10) * u.TeV
crab_flux = crab_magic_new(crab_energy)
crab_flux2 = crab_magic_2(crab_energy)

plt.figure(figsize=(8, 5))
flux_points_dataset = FluxPointsDataset(
    data=flux_points, models=model_best_joint
)
ax_pts, ax_res = flux_points_dataset.plot_fit();
ax_pts.plot(crab_energy, crab_flux.to(u.erg/(u.cm**2 * u.s)), color='red', label='MAGIC arXiv:1406.6892');
ax_pts.plot(crab_energy, crab_flux2.to(u.erg/(u.cm**2 * u.s)), color='blue', label='MAGIC arXiv:1406.6892');
plt.show()