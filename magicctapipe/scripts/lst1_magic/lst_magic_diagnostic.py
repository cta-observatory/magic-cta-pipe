"""
Usage:
After producing the MC gammas and protons DL2 files,
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
from magicctapipe.io import load_mc_dl2_data_file, get_stereo_events, telescope_combinations, get_dl2_mean
from matplotlib import gridspec
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 14})
from pyirf.benchmarks import angular_resolution, energy_bias_resolution
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.irf import effective_area_per_energy
import pandas as pd


gh_efficiency = 0.9
theta_efficiency = 0.8



def diagnostic_plots(config_IRF,target_dir):
    
    quality_cuts= f"(disp_diff_mean < {np.sqrt(0.05)})"
    irf_type= config_IRF["create_irf"]["event_type"]
    dl2_weight_type="intensity"
    energy_bins=np.logspace(-2,3,15)[2:]
    
    input_file_gamma = glob.glob(target_dir+'/DL2/MC/gammas/*gamma*.h5')
    input_file_gamma.sort()
    
    print(f"{len(input_file_gamma)} gamma files are found")

    input_file_proton = glob.glob(target_dir+'/DL2/MC/protons_test/*proton*.h5')
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
    plt.figure(figsize=(10,8),dpi=200)
    plt.xlabel("True energy of the simulated gamma rays [TeV]")
    plt.ylabel("Gammaness")
    plt.hist2d(x,y, bins=50, norm=mpl.colors.LogNorm())
    plt.colorbar(label="Number of events")
    plt.grid(linestyle=':')
    plt.title("Simulated gamma rays")
    plt.savefig(target_dir+"/gammaness_photons.png",bbox_inches='tight')


    x=np.array(background_hist['true_energy'].value)
    y=np.array(background_hist['gammaness'].value)
    plt.figure(figsize=(10,8),dpi=200)
    plt.xlabel("True energy of the simulated protons [TeV]")
    plt.ylabel("Gammaness")
    plt.hist2d(x,y, bins=50,  norm=mpl.colors.LogNorm())
    plt.colorbar(label="Number of events")
    plt.grid(linestyle=':')
    plt.title("Simulated protons")
    plt.savefig(target_dir+"/gammaness_protons.png",bbox_inches='tight')
    
    #Migration matrix
    x=np.array(np.log10(signal_hist['reco_energy'].value))
    y=np.array(np.log10(signal_hist['true_energy'].value))
    plt.figure(figsize=(10,8),dpi=200)
    plt.xlabel("Log Reconstructed energy (TeV)")
    plt.ylabel("Log True energy of the simulated photon (TeV)")
    plt.hist2d(x,y, bins=100,  norm=mpl.colors.LogNorm())
    plt.plot([-2,2],[-2,2],color="red")
    plt.ylim(y.min(),y.max())
    plt.xlim(x.min(),x.max())
    plt.colorbar(label="Number of events")
    plt.grid(linestyle=':')
    plt.savefig(target_dir+"/migration_matrix.png",bbox_inches='tight')
    
    #g-h separation
    gh_bins = np.linspace(0, 1, 51)
    plt.figure(figsize=(10,8),dpi=200)
    plt.xlabel("Gammaness")
    plt.ylabel("Number of events")
    plt.yscale("log")
    plt.grid(linestyle=':')
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
    
    
    #g-h separation per energy
    
    n_columns = 3
    n_rows = int(np.ceil(len(energy_bins[:-1]) / n_columns))

    grid = (n_rows, n_columns)
    locs = list(itertools.product(range(n_rows), range(n_columns)))

    plt.figure(figsize=(20, n_rows * 8),dpi=200)

    for i_bin, (eng_lolim, eng_uplim) in enumerate(zip(energy_bins[:-1], energy_bins[1:])):

        plt.subplot2grid(grid, locs[i_bin])
        plt.title(f"{eng_lolim:.3f} < energy < {eng_uplim:.3f} [TeV]", fontsize=20)
        plt.xlabel("Gammaness", fontsize=22)
        plt.yscale("log")
        plt.grid(linestyle=':')

        # Apply the energy cuts
        cond_back_lolim = background_hist["reco_energy"].value > eng_lolim
        cond_back_uplim = background_hist["reco_energy"].value < eng_uplim

        cond_signal_lolim = signal_hist["reco_energy"].value > eng_lolim
        cond_signal_uplim = signal_hist["reco_energy"].value < eng_uplim

        condition_back = np.logical_and(cond_back_lolim, cond_back_uplim)
        condition_signal = np.logical_and(cond_signal_lolim, cond_signal_uplim)

        dt_back = background_hist[condition_back]
        dt_signal = signal_hist[condition_signal]

        # Plot the background gammaness distribution
        if len(dt_back) > 0:
            plt.hist(
                 dt_back["gammaness"].value,
                 bins=gh_bins,
                 label="background",
                 histtype="step",
                 linewidth=2,
           )

        # Plot the signal gammaness distribution
        if len(dt_signal) > 0:
            plt.hist(
                dt_signal["gammaness"].value,
                bins=gh_bins,
                label="signal",
                histtype="step",
                linewidth=2,
            )

        plt.legend(loc="lower left")
    
    plt.savefig(target_dir+"/h-g_classification_per_energy.png",bbox_inches='tight')
    
    
    
    #Calculate the angular resolution
    
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
    
    
    
    mask_gh_eff_26 = evaluate_binned_cut(
        values=signal_hist_6_26["gammaness"],
        bin_values=signal_hist_6_26["reco_energy"],
        cut_table=gh_table_eff,
        op=operator.ge,
    )

    data_eff_gcut_26 = signal_hist_6_26[mask_gh_eff_26]


    angres_table_eff = angular_resolution(
        data_eff_gcut, u.Quantity(energy_bins, u.TeV), energy_type="reco"
    )

    angres_eff = angres_table_eff["angular_resolution"].value
    
    energy_bins_center = (energy_bins[:-1] + energy_bins[1:]) / 2
    energy_bins_width = [
        energy_bins[1:] - energy_bins_center,
        energy_bins_center - energy_bins[:-1],
    ]

    plt.figure(figsize=(10,8),dpi=200)
    gs = gridspec.GridSpec(4, 1)

    plt.title(f"angular resolution(g/h efficiency = {100*gh_efficiency}%)")
    plt.ylabel("Angular resolution (68% cont.) [deg]")
    plt.xlabel("Energy [TeV]")
    plt.semilogx()
    plt.ylim(0.07,0.165)
    plt.grid(linestyle=':')

    plt.errorbar(
        x=energy_bins_center,
        y=angres_eff,
        xerr=energy_bins_width,
        label="signal",
        marker="o",
    )


    plt.legend()
    plt.savefig(target_dir+"/ang_resolution.png",bbox_inches='tight')
    
    print("Type and values: ",type(energy_bins_center),energy_bins_center)
    print("Type and values: ",type(angres_eff),angres_eff)
    print("Type and values: ",type(energy_bins_width),energy_bins_width)
    
    np.savetxt("angular_resolution_data.txt",[energy_bins_center, angres_eff, energy_bins_width[0]],header="Energy_bin_center, ang_resolution, bin_x_width",delimiter=",")
    
    """
    # Dynamic gammaness cuts
    gh_cuts_eff = gh_table_eff["cut"].value
    print(f"Energy bins: {energy_bins}")
    print(f"Efficiency gammaness cuts:\n{gh_cuts_eff}")
    
    plt.figure(figsize=(10,8),dpi=200)
    plt.xlabel("Reconstructed energy [TeV]")
    plt.ylabel("Gammaness cut that saves 90% of the gamma rays")
    plt.semilogx()
    plt.grid(linestyle=':')

    plt.errorbar(
        x=energy_bins_center,
        y=gh_cuts_eff,
        xerr=energy_bins_width,
        label="gamma efficiency",
        marker="o",
    )
    """
    
    #Energy resolution
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

    plt.figure(figsize=(10,8),dpi=200)
    gs = gridspec.GridSpec(4, 1)

    plt.title(f"Energy bias and energy resolution(g/h eff. = {100*gh_efficiency}%, theta eff. = {100*theta_efficiency}%)")
    plt.ylabel("Energy bias and resolution")
    plt.xlabel("Reconstructed energy [TeV]")

    plt.semilogx()
    plt.grid(linestyle=':')

    plt.errorbar(
        x=energy_bins_center,
        y=engres_eff,
        xerr=energy_bins_width,
        label="Energy resolution",
        marker="o",
    )
    plt.ylim(-0.15,0.25)

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
    
    np.savetxt("energy_resolution_data.txt",[energy_bins_center, engres_eff, engbias_eff, energy_bins_width[0]],header="Energy_bin_center, ener_resolution, ener_bias, bin_x_width",delimiter=",")
    
    
    #Effective area
    mask_theta_eff_26 = evaluate_binned_cut(
        values=data_eff_gcut_26["theta"],
        bin_values=data_eff_gcut_26["reco_energy"],
        cut_table=theta_table_eff,
        op=operator.le,
    )

    data_eff_gtcuts_26 = data_eff_gcut_26[mask_theta_eff_26]
    
    aeff_eff_26 = effective_area_per_energy(
        selected_events=data_eff_gtcuts_26,
        simulation_info=sim_isto_signal,
        true_energy_bins=u.Quantity(energy_bins, u.TeV),
    )

    plt.figure(figsize=(10,8),dpi=200)
    plt.title(f"g/h eff. = {100*gh_efficiency}%, theta eff. = {100*theta_efficiency}%")
    plt.xlabel("Reconstructed energy [TeV]")
    plt.ylabel("Effective area [m$^2$]")
    plt.loglog()
    plt.grid(which="both",linestyle=':')
    plt.ylim(0.6e5,4e5)

    plt.errorbar(
        x=energy_bins_center,
        y=aeff_eff_26.value,
        xerr=energy_bins_width,
        label="zenith < 26$^{\circ}$",
        marker="o",
    )

    plt.legend(loc="lower right")
    plt.savefig(target_dir+"/effective_area.png",bbox_inches='tight')
    
    np.savetxt("effective_area.txt",[energy_bins_center, aeff_eff_26.value, energy_bins_width[0]],header="Energy_bin_center, effec_area, bin_x_width",delimiter=",")
    
    #Cmap (this will first check if you have real data)
    list_of_nights = np.sort(glob.glob(target_dir+'/DL2/Observations/*'))
    input_file_gamma = np.asarray([])
    if len(list_of_nights) > 0:
        for night in list_of_nights:
            input_file_gamma = np.concatenate([input_file_gamma, np.sort(glob.glob(night+'/*.h5'))])
    
    if len(input_file_gamma) > 0:
        _, TEL_COMBINATIONS = telescope_combinations(config_IRF)
        data_list = []
        for input_file in input_file_gamma:

            # Load the input file
            df_events = pd.read_hdf(input_file, key="events/parameters")
            data_list.append(df_events)

        event_data = pd.concat(data_list)
        event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
        event_data.sort_index(inplace=True)
        
        # Apply the quality cuts
        print(f"\nQuality cuts cmap: {quality_cuts}")
        event_data = get_stereo_events(event_data, config_IRF, quality_cuts)
        
        combo_type_Stereo_MAGIC = np.arange(len(TEL_COMBINATIONS))[-1]
        print("Excluding the MAGIC-stereo combination events...")
        event_data.query(f"combo_type != {combo_type_Stereo_MAGIC}", inplace=True)
        
        print("Calculating mean DL2 parameters...")
        event_data_mean = get_dl2_mean(event_data)
        
        # Only events observed with all 3 telescopes:
        
        combo_types = list(np.arange(len(TEL_COMBINATIONS))[0:-1]) # Here we use everything but MAGIC Stereo. We can actually use any combination, e.g.: [0,1,2], [1,2], [2] etc 
        # Only events observed with high prob of being gamma rays:
        cut_value_gh = 0.8

        print(f"Cmap combination types: {combo_types}")
        print(f"Global gammaness cut: {cut_value_gh}")

        # Get the photon list
        event_list = event_data_mean.query(
            f"(combo_type == {combo_types}) & (gammaness > {cut_value_gh})"
        ).copy()

        print(f"\nNumber of events: {len(event_list)}")
        
        plt.figure(dpi=200)
        plt.title(f"gammaness > {cut_value_gh}, combo_type = {combo_types}")
        plt.xlabel("RA [deg]")
        plt.ylabel("Dec [deg]")
        #plt.xlim(xlim)
        #plt.ylim(ylim)

        # Plot the counts map
        plt.hist2d(
            event_list["reco_ra"],
            event_list["reco_dec"],
            bins=[100,100], #np.linspace(xlim[1], xlim[0], 101), np.linspace(ylim[0], ylim[1], 101)],
        )

        plt.colorbar(label="Number of events")
        #plt.axis(xlim.tolist() + ylim.tolist())
        plt.grid(linestyle=':')

        plt.savefig(target_dir+"/counts_map.png",bbox_inches='tight')
    
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


    
    
    
