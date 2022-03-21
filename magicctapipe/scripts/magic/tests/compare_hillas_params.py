import pandas as pd 
import uproot
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import h5py
import yaml

from math import *
from params_for_hillas_comparison import mars_params, mcp_params, scale_factors, threshold, labels_and_units, scaling
from pathlib import Path

def compare_hillas_parameters(config_file ="config.yaml", hillas_key="dl1/hillas_params", subrun=5086952, plot_image=False):
	"""
	This fuction compares the values of the hillas parameters for MARS and magic-ctapipe. It returns a list of True/False 
	corresponding to the input hillas parameters.
	True : Error found, False : No Errors
	----------
	config_file: path of config file
	key: key in the h5 files (magic-cta-pipe output) where the hillas params are saved
	subrun: subrun number 
	plot_image: set to True if you want the scatterplot images saved
	"""
	
	config = yaml.safe_load(open(config_file, "r"))
	date = Path(config["MARS-input"]["MARS-path"]).name[:8]
	
	#--------------
	# read mcp data
	#--------------
	mcp_file = config["magic-cta-pipe-input"]["MCP-path"]

	df_mcp = pd.read_hdf(mcp_file, key=hillas_key)

	#filter for subrun
	df_mcp = (df_mcp.loc[df_mcp["obs_id"] == subrun])

	#apply cuts
	# 50<Size<50000, leakage<0.15
	df_mcp_cut = df_mcp.loc[(df_mcp["intensity"] < 50000) & (df_mcp["intensity"]>50) & (df_mcp["intensity_width_1"]<0.15)]


	# filter for M1 or M2
	df_mcp_m1 = df_mcp_cut[df_mcp_cut["tel_id"]==1]
	df_mcp_m2 = df_mcp_cut[df_mcp_cut["tel_id"]==2]

	#---------------
	# read MARS data
	#---------------
	# contains values for both M1 and M2
	cut_params = ["size_M1", "size_M2", "leakage_M1", "leakage_M2"]
	df_mars = pd.DataFrame()
	with uproot.open(config["MARS-input"]["MARS-path"]) as file_mars:
		for cut_par in cut_params:
			mars_name = mars_params[cut_par]
			df_mars["event_id"] = file_mars["Events"]["MRawEvtHeader_1.fStereoEvtNumber"].array(library ="np")
			df_mars["mars_"+cut_par] = file_mars["Events"][mars_name].array(library ="np")
		for par in config["Hillas-params"]["Params"]:
			mars_name = mars_params[par]
			df_mars["mars_"+par] = file_mars["Events"][mars_name].array(library ="np")

	# apply cuts 
	# 50<Size<50000, leakage<0.15
	df_mars_cut = df_mars.loc[(df_mars["mars_size_M1"] < 50000) & (df_mars["mars_size_M2"] < 50000) & (df_mars["mars_size_M1"] > 50) & 
							(df_mars["mars_size_M2"] > 50) & (df_mars["mars_leakage_M1"] < 0.15) & (df_mars["mars_leakage_M2"] < 0.15)]

	#--------------------------
	# compare hillas parameters
	#--------------------------
	df_merge_m1 = pd.merge(df_mars_cut, df_mcp_m1, on=["event_id"], how="inner")
	df_merge_m2 = pd.merge(df_mars_cut, df_mcp_m2, on=["event_id"], how="inner")
	comparison = []
	
	for par in config["Hillas-params"]["Params"]:
		df_hillas = pd.DataFrame()
		if par[-1] == "1":
			df_hillas["event_id"] = df_merge_m1["event_id"]
			df_hillas[par+"_mcp"] = df_merge_m1[mcp_params[par]]*scale_factors[par]
			df_hillas[par+"_mars"] = df_merge_m1["mars_"+par]
			
		elif par[-1] == "2":
			df_hillas["event_id"] = df_merge_m2["event_id"]
			df_hillas[par+"_mcp"] = df_merge_m2[mcp_params[par]]*scale_factors[par]
			df_hillas[par+"_mars"] = df_merge_m2["mars_"+par]
			
		df_hillas["relative_error"]=(df_hillas[par+"_mars"]-df_hillas[par+"_mcp"])/df_hillas[par+"_mcp"]
		error = df_hillas.loc[(df_hillas["relative_error"]>threshold[par])].to_numpy()

		if error.size <= len(df_hillas)*0.01: #percentage of the events, that is allowed to have errors bigger than the specified threshold
			errors_found = False
		else:
			errors_found = True
			# df_hillas.to_hdf("{}/{}_hillas_{}.h5".format(config["Output_paths"]["file_output_directory"], subrun, par), "/hillas_params", "w")
			df_hillas.to_hdf("{}/{}_hillas_comparison.h5".format(config["Output_paths"]["file_output_directory"], subrun), f"/{par}", "a")
		comparison.append(errors_found)
	
		#----------------------
		# plot image (optional)
		#----------------------
		#scaling factors are already taken into account in df_hillas
		if plot_image == True:
			if scaling[par] == "log":
				plt.hist2d(df_hillas[par+"_mcp"], df_hillas[par+"_mars"], bins=np.logspace(np.log10(10), np.log10(10e5), 300), 
				norm=colors.LogNorm(), cmap=plt.cm.jet, cmin=1) 	
				plt.loglog()
			elif scaling[par] == "lin":
				plt.hist2d(df_hillas[par+"_mcp"], df_hillas[par+"_mars"], bins=300, norm=colors.LogNorm(), cmap=plt.cm.jet, cmin=1)
			plt.colorbar()
			plt.xlabel("magic-cta-pipe")
			plt.ylabel("MARS")
			plt.title(f"{date}_{subrun}_S - {labels_and_units[par]}")
			xpoints = ypoints = plt.xlim()
			plt.plot(xpoints, ypoints, linestyle="--", color="y", lw=1, scalex=False, scaley=False)
			plt.savefig("{}/{}_hillas_comparison_{}.png".format(config["Output_paths"]["image_output_directory"], subrun, par), facecolor="w", transparent=False)
			plt.close()
	return comparison
	