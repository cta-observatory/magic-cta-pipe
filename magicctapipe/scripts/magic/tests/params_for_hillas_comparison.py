import math
mars_params = {
	"event_id" : "MRawEvtHeader_1.fStereoEvtNumber",
	"length_M1" : "MHillas_1.fLength", "length_M2" : "MHillas_2.fLength", 
	"width_M1" : "MHillas_1.fWidth", "width_M2" : "MHillas_2.fWidth", 
	"size_M1" : "MHillas_1.fSize", "size_M2" : "MHillas_2.fSize", 
	"slope_M1" : "MHillasTimeFit_1.fP1Grad", "slope_M2" : "MHillasTimeFit_2.fP1Grad", 
	"delta_M1" : "MHillas_1.fDelta", "delta_M2" : "MHillas_2.fDelta", 
	"cogx_M1" : "MHillas_1.fMeanX", "cogx_M2" : "MHillas_2.fMeanX", 
	"cogy_M1" : "MHillas_1.fMeanY", "cogy_M2" : "MHillas_2.fMeanY", 
	# "hmax" : "MStereoPar.fMaxHeight", 
	# "corex" : "MStereoPar.fCoreX", 
	# "corey" : "MStereoPar.fCoreY", 
	# "az" : "MStereoPar.fDirectionAz", 
	# "zd" : "MStereoPar.fDirectionZd",
	# "n_islands" : "MImagePar_1./MImagePar_1.fNumIslands",
	"leakage_M1" : "MNewImagePar_1./MNewImagePar_1.fLeakage1", "leakage_M2" : "MNewImagePar_2./MNewImagePar_2.fLeakage1"
}
mcp_params = {
	"length_M1" : "length", "length_M2" : "length", 
	"width_M1" : "width", "width_M2" : "width",
	"size_M1" : "intensity", "size_M2" : "intensity", 
	"slope_M1" : "slope", "slope_M2" : "slope", 
	"delta_M1" : "psi", "delta_M2" : "psi", 
	"cogx_M1" : "x", "cogx_M2" : "x", 
	"cogy_M1" : "y", "cogy_M2" : "y", 
	# "hmax" : "h_max", 
	# "corex" : "core_x", 
	# "corey" : "core_y", 
	# "az" : "az", 
	# "zd" : "alt",
	# "n_islands" : "n_islands",
	"leakage_M1" : "intensity_width_1", "leakage_M2" : "intensity_width_1"
}

scale_factors = { 
	"length_M1" : 1000., "length_M2" : 1000., 
	"width_M1" : 1000., "width_M2" : 1000.,
	"size_M1" : 1., "size_M2" : 1., 
	"slope_M1" : 1., "slope_M2" : 1., 
	"delta_M1" : (math.pi*2.0)/360., "delta_M2" : (math.pi*2.0)/360., 
	"cogx_M1" : 1000., "cogx_M2" : 1000., 
	"cogy_M1" : 1000., "cogy_M2" : 1000., 
	"slope_M1" : 0.001, "slope_M2" : 0.001, 
	"leakage_M1" : 1., "leakage_M2" : 1.,
	# "hmax" : 100.,
	# "corex" : 100., 
	# "corey" : 100.,
	# "az" : 1.,
	# "zd" : 1.,
	# "n_islands" : 1.
	}
#important for the plotting, these are mars units
labels_and_units = {
	"length_M1" : "Length M1 [mm]", "length_M2" : "Length M2 [mm]", 
	"width_M1" : "Width M1 [mm]", "width_M2" : "Width M2 [mm]", 
	"size_M1" : "Size M1 [phe]", "size_M2" : "Size M2 [phe]", 
	"delta_M1" : "Delta M1 [rad]", "delta_M2" : "Delta M2 [rad]", 
	"slope_M1" : "Time Gradient M1", "slope_M2" : "Time Gradient M2", 
	"cogx_M1" : "Cog_x M1 [mm]", "cogx_M2" : "Cog_x M2 [mm]", 
	"cogy_M1" : "Cog_y M1 [mm]", "cogy_M2" : "Cog_y M2 [mm]", 
	# "hmax" : "Max height [cm]", 
	# "corex" : "CoreX [cm]", 
	# "corey" : "CoreY [cm]", 
	# "az" : "Azimuth [deg]", 
	# "zd" : "Zenith [deg]",
	"leakage_M1" : "Leakage", "leakage_M2" : "Leakage"
}
#threshold for relative error
threshold = {
	"length_M1" : 0.01, "length_M2" : 0.01,
	"width_M1" : 0.01, "width_M2" : 0.01, 
	"size_M1" : 0.01, "size_M2" : 0.01, 
	"delta_M1" : 0.01, "delta_M2" : 0.01, 
	"slope_M1" : 1, "slope_M2" : 1, 
	"cogx_M1" : 0.01, "cogx_M2" : 0.01, 
	"cogy_M1" : 0.01, "cogy_M2" : 0.01, 
	# "hmax" : 1, 
	# "corex" : 1, 
	# "corey" : 1, 
	# "az" : 1, 
	# "zd" : 1,
	"leakage_M1" : 0.01, "leakage_M2" : 0.01
}

scaling = {
	"length_M1" : "lin", "length_M2" : "lin", 
	"width_M1" : "lin", "width_M2" : "lin", 
	"size_M1" : "log", "size_M2" : "log", 
	"delta_M1" : "lin", "delta_M2" : "lin", 
	"slope_M1" : "lin", "slope_M2" : "lin", 
	"cogx_M1" : "lin", "cogx_M2" : "lin", 
	"cogy_M1" : "lin", "cogy_M2" : "lin", 
	"leakage_M1" : "lin", "leakage_M2" : "lin"
	# "hmax" : "lin", 
	# "corex" : "lin", 
	# "corey" : "lin", 
	# "az" : "lin", 
	# "zd" : "lin"
}
