import pandas as pd 
import uproot
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import yaml
from astropy.coordinates import angular_separation, EarthLocation, AltAz, SkyCoord
from astropy.time import Time
from astropy import units as u
from math import *
from labels_and_units_for_params_comparison import mars_params, mcp_params, scale_factors, threshold, labels_and_units, scaling
from pathlib import Path


def compare_hillas_stereo_parameters(config_file="config.yaml", params_key="events/parameters", plot_image=False):
    """
    This fuction compares the values of the hillas and stereo parameters for MARS and magic-ctapipe. It returns a list of True/False
    corresponding to the input compared parameters.
    True : Error found, False : No Errors
    ----------
    config_file: path of config file
    params_key: key in the dl1_stereo files (magic-cta-pipe output) where the hillas and stereo params are saved
    plot_image: if True, a plot is created for each parameter and saved
    """

    config = yaml.safe_load(open(config_file, "r"))
    filename_no_ext = Path(config["MARS-input"]["MARS-path"]).stem

    # --------------
    # read mcp data
    # --------------
    mcp_file = config["magic-cta-pipe-input"]["MCP-path"]

    df_mcp = pd.read_hdf(mcp_file, key=params_key)

    # filter for subrun
    #df_mcp = df_mcp.loc[df_mcp["obs_id"] == subrun]

    print(f"Number of events before cuts {df_mcp.shape[0]}")

    LON_ORM = u.Quantity(-17.89064, u.deg)
    LAT_ORM = u.Quantity(28.76177, u.deg)
    HEIGHT_ORM = u.Quantity(2199.835, u.m)

    # location = EarthLocation.from_geodetic(lon=LON_ORM, lat=LAT_ORM, height=HEIGHT_ORM)
    # times1 = [str(float(sec))[:-2] for sec in df_mcp['time_sec']]
    # times2 = [str(float(nano))[:-2] for nano in df_mcp['time_nanosec']]
    # times = [f"{sec}.{nanosec}" for sec, nanosec in zip(times1, times2)]
#
    # event_times = Time(times, format="unix", scale="utc")
    # source_coords = SkyCoord(ra=83.6333 * u.degree, dec=22.0133 * u.degree, frame='icrs')
    # la_palma = AltAz(location=location, obstime=event_times)
    # source_coords_altaz = source_coords.transform_to(la_palma)
#
    # theta = angular_separation(
        # lon1=u.Quantity(df_mcp['az'].to_numpy(), u.deg),
        # lat1=u.Quantity(df_mcp['alt'].to_numpy(), u.deg),
        # lon2=source_coords_altaz.az,
        # lat2=source_coords_altaz.alt,
    # )
#
    # df_mcp['theta2'] = theta.to(u.deg).value ** 2
    # df_mcp['theta'] = theta.to(u.deg).value

    # apply cuts
    # 50<Size<50000, leakage<0.15, impact<120, theta<0.14
    # df_mcp_cut = df_mcp.loc[(df_mcp["intensity"] < 50000) & (df_mcp["intensity"] > 50) & (df_mcp["intensity_width_1"] < 0.15) & (df_mcp['theta'] < 0.14) & (df_mcp["impact"] < 120)]
    df_mcp_cut = df_mcp.loc[(df_mcp["intensity"] < 50000) & (df_mcp["intensity"] > 50) & (df_mcp["intensity_width_1"] < 0.15)]

    # filter for M1 or M2
    df_mcp_m1 = df_mcp_cut[df_mcp_cut["tel_id"] == 2]
    df_mcp_m2 = df_mcp_cut[df_mcp_cut["tel_id"] == 3]

    print(f"NUmber of M1 events after cut {df_mcp_m1.shape[0]}")
    print(f"NUmber of M2 events after cut {df_mcp_m2.shape[0]}")

    # ---------------
    # read MARS data
    # ---------------
    # contains values for both M1 and M2
    cut_params = ["size_M1", "size_M2", "leakage_M1", "leakage_M2", "impact_M1", "impact_M2"]
    df_mars = pd.DataFrame()
    with uproot.open(config["MARS-input"]["MARS-path"]) as file_mars:
        for cut_par in cut_params:
            mars_name = mars_params[cut_par]
            df_mars["event_id"] = file_mars["Events"]["MRawEvtHeader_1.fStereoEvtNumber"].array(library="np")
            df_mars["mars_" + cut_par] = file_mars["Events"][mars_name].array(library="np")
        for par in config["Params"]:
            mars_name = mars_params[par]
            df_mars["mars_" + par] = file_mars["Events"][mars_name].array(library="np")

    # apply cuts
    # 50<Size<50000, leakage<0.15, impact<120
    df_mars_cut = df_mars.loc[(df_mars["mars_size_M1"] < 50000) & (df_mars["mars_size_M2"] < 50000) & (df_mars["mars_size_M1"] > 50) & \
                            (df_mars["mars_size_M2"] > 50) & (df_mars["mars_leakage_M1"] < 0.15) & (df_mars["mars_leakage_M2"] < 0.15) & \
                            (df_mars["mars_impact_M1"] < 12000) & (df_mars["mars_impact_M2"] < 12000)]
    # --------------------------
    # compare hillas and stereo parameters
    # --------------------------
    df_merge_m1 = pd.merge(df_mars_cut, df_mcp_m1, on=["event_id"], how="inner")
    df_merge_m2 = pd.merge(df_mars_cut, df_mcp_m2, on=["event_id"], how="inner")
    print(f"NUmber of M1 events after merging {df_merge_m1.shape[0]}")
    print(f"NUmber of M2 events after merging {df_merge_m2.shape[0]}")
    comparison = []

    for par in config["Params"]:
        df_params = pd.DataFrame()
        if par == "zd":
            df_params["event_id"] = df_merge_m2["event_id"]
            df_params[par+"_mcp"] = 90 - df_merge_m2[mcp_params[par]]*scale_factors[par]
            df_params[par+"_mars"] = df_merge_m2["mars_"+par]
            df_params = df_params.drop(df_params[np.isnan(df_params[par+"_mars"])].index)
        elif par == "az":
            df_params["event_id"] = df_merge_m2["event_id"]
            df_params[par+"_mcp"] = df_merge_m2[mcp_params[par]]*scale_factors[par]
            df_params[par+"_mars"] = df_merge_m2["mars_"+par]
            df_params = df_params.drop(df_params[np.isnan(df_params[par+"_mars"])].index)
            for az in df_params[par+"_mars"]:
                if az < 0:
                    df_params[par+"_mars"] = df_params[par+"_mars"].replace(az, az+360)
        elif par == "delta_M1":
            df_params["event_id"] = df_merge_m1["event_id"]
            df_params[par+"_mcp"] = np.pi/2 - df_merge_m1[mcp_params[par]]*scale_factors[par]
            df_params[par+"_mars"] = df_merge_m1["mars_"+par]
            for delta in df_params[par+"_mars"]:
                if delta < 0:
                    df_params[par+"_mars"] = df_params[par+"_mars"].replace(delta, delta+np.pi)
        elif par == "delta_M2":
            df_params["event_id"] = df_merge_m2["event_id"]
            df_params[par+"_mcp"] = np.pi/2 - df_merge_m2[mcp_params[par]]*scale_factors[par]
            df_params[par+"_mars"] = df_merge_m2["mars_"+par]
            for delta in df_params[par+"_mars"]:
                if delta < 0:
                    df_params[par+"_mars"] = df_params[par+"_mars"].replace(delta, delta+np.pi)
        elif par == "hmax":
            df_params["event_id"] = df_merge_m2["event_id"]
            df_params[par+"_mcp"] = df_merge_m2[mcp_params[par]]*scale_factors[par]
            df_params[par+"_mars"] = df_merge_m2["mars_"+par]
            df_params = df_params.drop(df_params[df_params[par+"_mars"] < 0].index)
        #   df_params = df_params.drop(df_params[np.isnan(df_params[par+"_mars"])].index)
        elif par == "slope_M1":
            df_params["event_id"] = df_merge_m2["event_id"]
            df_params[par+"_mcp"] = abs(df_merge_m2[mcp_params[par]]*scale_factors[par])
            df_params[par+"_mars"] = abs(df_merge_m2["mars_"+par])
        elif par == "slope_M2":
            df_params["event_id"] = df_merge_m2["event_id"]
            df_params[par+"_mcp"] = abs(df_merge_m2[mcp_params[par]]*scale_factors[par])
            df_params[par+"_mars"] = abs(df_merge_m2["mars_"+par])

        elif par[-1] == "1":
            df_params["event_id"] = df_merge_m1["event_id"]
            df_params[par+"_mcp"] = df_merge_m1[mcp_params[par]]*scale_factors[par]
            df_params[par+"_mars"] = df_merge_m1["mars_"+par]
        elif par[-1] == "2":
            df_params["event_id"] = df_merge_m2["event_id"]
            df_params[par+"_mcp"] = df_merge_m2[mcp_params[par]]*scale_factors[par]
            df_params[par+"_mars"] = df_merge_m2["mars_"+par]
        else:
            df_params["event_id"] = df_merge_m2["event_id"]
            df_params[par+"_mcp"] = df_merge_m2[mcp_params[par]]*scale_factors[par]
            df_params[par+"_mars"] = df_merge_m2["mars_"+par]

        df_params["relative_error"]=(df_params[par+"_mars"] - df_params[par+"_mcp"])/df_params[par+"_mcp"]
        error = df_params.loc[(df_params["relative_error"]>threshold[par])].to_numpy()

        Path(config["Output_paths"]["file_output_directory"]).mkdir(exist_ok=True, parents=True)
        Path(config["Output_paths"]["image_output_directory"]).mkdir(exist_ok=True, parents=True)

        if error.size <= len(df_params) * 0.001:  # percentage of the events, that is allowed to have errors bigger than the specified threshold
            errors_found = False
        else:
            df_params.to_hdf(f'{config["Output_paths"]["file_output_directory"]}/{filename_no_ext}_hillas_comparison.h5', "a")
            errors_found = True
        comparison.append(errors_found)

        # ----------------------
        # plot image (optional)
        # ----------------------
        # scaling factors are already taken into account in df_params
        if plot_image is True:
            if scaling[par] == "log":
                plt.hist2d(df_params[par + "_mcp"], df_params[par + "_mars"], bins=np.logspace(np.log10(10), np.log10(10e5), 300),
                norm=colors.LogNorm(), cmap=plt.cm.jet, cmin=1)
                plt.loglog()
            elif scaling[par] == "lin":
                plt.hist2d(df_params[par + "_mcp"], df_params[par + "_mars"], bins=[300, 300], norm=colors.LogNorm(), cmap=plt.cm.jet, cmin=1)
            plt.colorbar()
            plt.xlabel("magic-cta-pipe")
            plt.ylabel("MARS")
            if "_S_" in config["MARS-input"]["MARS-path"]:
                title_substr = "_S"
            elif "_Q_" in config["MARS-input"]["MARS-path"]:
                title_substr = "_Q"
            else:
                title_substr = ""
            plt.title(f"{filename_no_ext}{title_substr} - {labels_and_units[par]}")
            xpoints = ypoints = plt.xlim()
            plt.plot(xpoints, ypoints, linestyle="--", color="y", lw=1, scalex=False, scaley=False)
            plt.savefig(f'{config["Output_paths"]["image_output_directory"]}/{filename_no_ext}_hillas_comparison_{par}.png', facecolor="w", transparent=False)
            plt.close()
    return comparison
