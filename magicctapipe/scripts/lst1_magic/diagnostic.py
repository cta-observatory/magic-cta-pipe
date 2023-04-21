"""
Usage:
After producing the gammas and protons DL2 files,
you can run this script on your workspace directory
(i.e., the one given in the "workspace_dir" entry in
the "config_general.yaml" file), by doing:

$ conda run -n magic-lst python diagnostic.py

Note that the file "config_general.yaml" must be in
the same directory as this script.

"""

import itertools
import operator
import glob
import yaml
import numpy as np
from astropy import units as u
from astropy.table import Table, QTable, vstack
from magicctapipe.io import load_mc_dl2_data_file
from matplotlib import gridspec
import matplotlib as mpl
from matplotlib import pyplot as plt
from pyirf.benchmarks import angular_resolution, energy_bias_resolution
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.irf import effective_area_per_energy


def diagnostic_plots(config_IRF,target_dir):
    
    quality_cuts= f"(disp_diff_mean < {np.sqrt(0.05)})"
    irf_type= config_IRF["create_irf"]["event_type"]
    dl2_weight_type="intensity"
    energy_bins=np.logspace(-2,3,15)[2:]
    
    input_file_gamma = glob.glob(target_dir+'/DL2/MC/*gamma*.h5')
    input_file_gamma.sort()
    
    print(f"{len(input_file_gamma)} gamma files are found")

    input_file_proton = glob.glob(target_dir+'/DL2/MC/*proton*.h5')
    input_file_proton.sort()
    
    print(f"{len(input_file_proton)} proton files are found")
    
    print("Loading the input files...")

    signal_hist=[]
    background_hist=[]
    signal_hist_6_26=[]
    signal_hist_26_46=[]
    signal_hist_46_67=[]

    #First we do for the gammas:
    for i_file, input_file in enumerate(input_file_gamma):
        # Load the input file
        sig_hist, point_sig, sim_isto_signal =load_mc_dl2_data_file(
            config_IRF, input_file, quality_cuts, irf_type, dl2_weight_type
        )
        
        if point_sig[0]<=26:
            signal_hist_6_26=vstack([signal_hist_6_26,sig_hist])
        elif point_sig[0]<=46:
            signal_hist_26_46=vstack([signal_hist_26_46,sig_hist])
        elif point_sig[0]<=67:
            signal_hist_46_67=vstack([signal_hist_46_67,sig_hist])
        signal_hist=vstack([signal_hist,sig_hist])

    #And then for the protons:
    for i_file, input_file in enumerate(input_file_proton):
        # Load the input file
        back_hist, point_back, sim_isto_back = load_mc_dl2_data_file(
            config_IRF, input_file, quality_cuts, irf_type, dl2_weight_type
       
        ) 
        
        background_hist=vstack([background_hist,back_hist])
    
    #gammaness:
    x=np.array(signal_hist['true_energy'].value)
    y=np.array(signal_hist['gammaness'].value)
    plt.figure()
    plt.xlabel("True energy of the simulated gamma rays [TeV]")
    plt.ylabel("Gammaness")
    plt.hist2d(x,y, bins=50, norm=mpl.colors.LogNorm())
    plt.colorbar(label="Number of events")
    plt.title("Simulated gamma rays")
    plt.savefig(target_dir+"/gammaness_photons.png",bbox_inches='tight')


    x=np.array(background_hist['true_energy'].value)
    y=np.array(background_hist['gammaness'].value)
    plt.figure()
    plt.xlabel("True energy of the simulated protons [TeV]")
    plt.ylabel("Gammaness")
    plt.hist2d(x,y, bins=50,  norm=mpl.colors.LogNorm())
    plt.colorbar(label="Number of events")
    plt.title("Simulated protons")
    plt.savefig(target_dir+"/gammaness_protons.png",bbox_inches='tight')
    
    #Migration matrix
    x=np.array(np.log10(signal_hist['reco_energy'].value))
    y=np.array(np.log10(signal_hist['true_energy'].value))
    plt.figure()
    plt.xlabel("Reconstructed energy (TeV)")
    plt.ylabel("True energy of the simulated photon (TeV)")
    plt.hist2d(x,y, bins=100,  norm=mpl.colors.LogNorm())
    plt.plot([-2,2],[-2,2],color="red")
    plt.ylim(y.min(),y.max())
    plt.xlim(x.min(),x.max())
    plt.colorbar(label="Number of events");
    plt.savefig(target_dir+"/migration_matrix.png",bbox_inches='tight')
    
    #g-h separation
    gh_bins = np.linspace(0, 1, 51)
    plt.figure(dpi=70)
    plt.xlabel("Gammaness")
    plt.ylabel("Number of events")
    plt.yscale("log")
    plt.grid()
    g_back=np.array(background_hist["gammaness"].value)
    g_sig=np.array(signal_hist["gammaness"].value)
    plt.hist(
        g_back,
        bins=gh_bins,
        label="protons (background)",
        histtype="step",
        linewidth=2,
    )
    plt.hist(
        g_sig,
        bins=gh_bins,
        label="gammas (signal)",
        histtype="step",
        linewidth=2,
    )

    plt.legend(loc="upper left")
    plt.savefig(target_dir+"/h-g_classification.png",bbox_inches='tight')
    
    
    # Calculate the angular resolution
    gh_efficiency = 0.9
    gh_percentile = 100 * (1 - gh_efficiency)

    gh_table_eff = calculate_percentile_cut(
        values=signal_hist["gammaness"],
        bin_values=signal_hist["reco_energy"],
        bins=u.Quantity(energy_bins, u.TeV),
        fill_value=0.0,
        percentile=gh_percentile,
    )

    mask_gh_eff = evaluate_binned_cut(
        values=signal_hist["gammaness"],
        bin_values=signal_hist["reco_energy"],
        cut_table=gh_table_eff,
        op=operator.ge,
    )
    data_eff_gcut = signal_hist[mask_gh_eff]

    angres_table_eff = angular_resolution(
        data_eff_gcut, u.Quantity(energy_bins, u.TeV), energy_type="reco"
    )

    angres_eff = angres_table_eff["angular_resolution"].value
    
    energy_bins_center = (energy_bins[:-1] + energy_bins[1:]) / 2
    energy_bins_width = [
        energy_bins[1:] - energy_bins_center,
        energy_bins_center - energy_bins[:-1],
    ]

    plt.figure(dpi=70)
    gs = gridspec.GridSpec(4, 1)

    plt.title(f"angular resolution(g/h efficiency = {100*gh_efficiency}%)")
    plt.ylabel("Angular resolution (68% cont.) [deg]")
    plt.xlabel("Energy [TeV]")
    plt.semilogx()
    plt.grid()

    # Plot the angular resolution
    plt.errorbar(
        x=energy_bins_center,
        y=angres_eff,
        xerr=energy_bins_width,
        label="signal",
        marker="o",
    )


    plt.legend()
    plt.savefig(target_dir+"/ang_resolution.png",bbox_inches='tight')

    #Energy resolution
    theta_efficiency = 0.8
    theta_percentile = 100 * theta_efficiency
    theta_table_eff = calculate_percentile_cut(
        values=data_eff_gcut["theta"],
        bin_values=data_eff_gcut["reco_energy"],
        bins=u.Quantity(energy_bins, u.TeV),
        fill_value=data_eff_gcut["theta"].unmasked.max(),
        percentile=theta_percentile,
    )
    mask_theta_eff = evaluate_binned_cut(
        values=data_eff_gcut["theta"],
        bin_values=data_eff_gcut["reco_energy"],
        cut_table=theta_table_eff,
        op=operator.le,
    )

    data_eff_gtcuts = data_eff_gcut[mask_theta_eff]
    engres_table_eff = energy_bias_resolution(
        data_eff_gtcuts, u.Quantity(energy_bins, u.TeV), energy_type="reco"
    )

    engbias_eff = engres_table_eff["bias"].value
    engres_eff = engres_table_eff["resolution"].value

    plt.figure()
    gs = gridspec.GridSpec(4, 1)

    plt.title(f"Energy bias and energy resolution(g/h eff. = {100*gh_efficiency}%, theta eff. = {100*theta_efficiency}%)")
    plt.ylabel("Energy bias and resolution")
    plt.xlabel("Reconstructed energy [TeV]")

    plt.semilogx()
    plt.grid()

    # Plot the signal energy bias and resolution
    plt.errorbar(
        x=energy_bins_center,
        y=engres_eff,
        xerr=energy_bins_width,
        label="Energy resolution",
        marker="o",
    )

    plt.errorbar(
        x=energy_bins_center,
        y=engbias_eff,
        xerr=energy_bins_width,
        label="Energy bias",
        marker="o",
        linestyle="--",
    )

    plt.legend()
    plt.savefig(target_dir+"/energy_resolution.png",bbox_inches='tight')
    
def main():

    """
    Here we read the config_general.yaml file and call the functions defined above.
    """
    
    
    with open("config_general.yaml", "rb") as f:   # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    
    target_dir = config["directories"]["workspace_dir"]+config["directories"]["target_name"]
    
    
    with open(target_dir+"/config_IRF.yaml", "rb") as f:   # "rb" mode opens the file in binary format for reading
        config_IRF = yaml.safe_load(f)
    
    diagnostic_plots(config_IRF,target_dir)
    

if __name__ == "__main__":
    main()


    
    
    
