import pandas as pd
import uproot
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import yaml
from pathlib import Path
from labels_and_units_for_params_comparison import (
    mars_params,
    mcp_params,
    scale_factors,
    threshold,
    labels_and_units,
    scaling,
)


def compare_slope(
    bins, config_file="config.yaml", params_key="events/parameters", plot_image=False,
):

    config = yaml.safe_load(open(config_file, "r"))
    filename_no_ext = Path(config["MARS-input"]["MARS-path"]).stem

    # read MCP data
    mcp_file = config["magic-cta-pipe-input"]["MCP-path"]

    # apply cuts
    # 50<Size<50000, leakage<0.15, impact<120, theta<0.14
    df_mcp = pd.read_hdf(mcp_file, key=params_key)
    df_mcp_cut = df_mcp.loc[
            (df_mcp["intensity"] < 50000)
            & (df_mcp["intensity"] > 50)
            & (df_mcp["intensity_width_1"] < 0.15)]

    # read MARS data
    cut_params = [
            "size_M1",
            "size_M2",
            "leakage_M1",
            "leakage_M2",
            "impact_M1",
            "impact_M2",
        ]
    df_mars = pd.DataFrame()
    with uproot.open(config["MARS-input"]["MARS-path"]) as file_mars:
        for cut_par in cut_params:
            mars_name = mars_params[cut_par]
            df_mars["event_id"] = file_mars["Events"][
                "MRawEvtHeader_1.fStereoEvtNumber"
            ].array(library="np")
            df_mars["mars_" + cut_par] = file_mars["Events"][mars_name].array(
                library="np"
            )
        for par in config["Params"]:
            mars_name = mars_params[par]
            df_mars["mars_" + par] = file_mars["Events"][mars_name].array(library="np")

    # apply cuts on MARS data
    # 50<Size<50000, leakage<0.15, impact<120
    df_mars_cut = df_mars.loc[
        (df_mars["mars_size_M1"] < 50000)
        & (df_mars["mars_size_M2"] < 50000)
        & (df_mars["mars_size_M1"] > 50)
        & (df_mars["mars_size_M2"] > 50)
        & (df_mars["mars_leakage_M1"] < 0.15)
        & (df_mars["mars_leakage_M2"] < 0.15)
        # & (df_mars["mars_impact_M1"] < 12000) 
        # & (df_mars["mars_impact_M2"] < 12000)
    ]
    comparison_list = []

    # iterate over n_pixels bins
    for item in bins.items():
        if len(item[1]) <= 3:   #minimum number of events required for comparison
            continue
        print(f"now comparing the slope for {item[0]}")

        df_mcp_cut = df_mcp.loc[
            (df_mcp["event_id"].isin(item[1]))
            & (df_mcp["tel_id"] == int(item[0][-1])+1)
            ]

        print(f"Number of events after cut {df_mcp_cut.shape[0]}")

        df_merge = pd.merge(df_mars_cut, df_mcp_cut, on=["event_id"], how="inner")
        print(f"Number of events after merging: {df_merge.shape[0]}")
        if df_merge.shape[0] <= 3:   #minimum number of events required for comparison
            continue

        for par in config["Params"]:
            comparison = []
            comparison_fraction = []
            df_params = pd.DataFrame()
            if par == "slope_M1":
                df_params["event_id"] = df_merge["event_id"]
                df_params[par + "_mcp"] = abs(
                    df_merge[mcp_params[par]] * scale_factors[par]
                )
                df_params[par + "_mars"] = abs(df_merge["mars_" + par])
            elif par == "slope_M2":
                df_params["event_id"] = df_merge["event_id"]
                df_params[par + "_mcp"] = abs(
                    df_merge[mcp_params[par]] * scale_factors[par]
                )
                df_params[par + "_mars"] = abs(df_merge["mars_" + par])
            else:
                df_params["event_id"] = df_merge_m2["event_id"]
                df_params[par + "_mcp"] = df_merge_m2[mcp_params[par]] * scale_factors[par]
                df_params[par + "_mars"] = df_merge_m2["mars_" + par]

            df_params["relative_error"] = (
                df_params[par + "_mars"] - df_params[par + "_mcp"]
            ) / df_params[par + "_mcp"]
            error = df_params.loc[
                (np.abs(df_params["relative_error"]) > threshold[par])
            ].to_numpy()

            Path(config["Output_paths"]["file_output_directory"]).mkdir(
                exist_ok=True, parents=True
            )
            Path(config["Output_paths"]["image_output_directory"]).mkdir(
                exist_ok=True, parents=True
            )

            if (
                len(error) <= len(df_params) * 0.01
            ):  # percentage of the events, that is allowed to have errors bigger than the specified threshold
                errors_found = False
            else:
                with pd.HDFStore(
                    f'{config["Output_paths"]["file_output_directory"]}/{filename_no_ext}_{item[0]}_{par}_comparison.h5'
                ) as store:
                    store.put(f"/{par}", df_params, format="table", data_columns=True)
                errors_found = True

            comparison.append(errors_found)
            comparison_fraction.append(len(error) / len(df_params))

            # ----------------------
            # plot image (optional)
            # ----------------------
            # scaling factors are already taken into account in df_params
            if plot_image is True:
                #plot histogram with colorbar
                if scaling[par] == "log":
                    plt.hist2d(
                        df_params[par + "_mcp"],
                        df_params[par + "_mars"],
                        bins=np.logspace(np.log10(10), np.log10(10e5), 300),
                        norm=colors.LogNorm(),
                        cmap=plt.cm.jet,
                        cmin=1,
                    )
                    plt.loglog()
                elif scaling[par] == "lin":
                    plt.hist2d(
                        df_params[par + "_mcp"],
                        df_params[par + "_mars"],
                        bins=[300, 10000],
                        norm=colors.LogNorm(),
                        cmap=plt.cm.jet,
                        cmin=1,
                    )
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
                plt.plot(
                    xpoints,
                    ypoints,
                    linestyle="--",
                    color="y",
                    lw=1,
                    scalex=False,
                    scaley=False,
                )
                plt.savefig(
                    f'{config["Output_paths"]["image_output_directory"]}/{filename_no_ext}_{par}_{item[0][:-3]}.png',
                    facecolor="w",
                    transparent=False,
                )
                plt.close()

                # plot scatterplot
                df_params.plot(kind='scatter',x=f'{par}_mcp',y=f'{par}_mars',color='red', 
                    title =f"{filename_no_ext}{title_substr} - {labels_and_units[par]}" )
                plt.plot(
                    xpoints,
                    ypoints,
                    linestyle="--",
                    color="y",
                    lw=1,
                    scalex=False,
                    scaley=False,
                )
                plt.savefig(
                        f'{config["Output_paths"]["image_output_directory"]}/{filename_no_ext}_{par}_scatter_{item[0][:-3]}.png',
                        facecolor="w",
                        transparent=False,
                    )
                # plot difference distribution
                plt.figure()
                plt.hist(df_params[par + "_mcp"] - df_params[par + "_mars"], bins=100)
                plt.xlabel("Value in MCP - Value in MARS")
                plt.suptitle(f"{filename_no_ext}{title_substr} - {labels_and_units[par]}")
                plt.title("Difference between slope values")
                plt.savefig(
                        f'{config["Output_paths"]["image_output_directory"]}/{filename_no_ext}_{item[0]}_{par}_comparison_distribution_{par}.png',
                        facecolor="w",
                        transparent=False,
                    )
                comparison_list.append((comparison, comparison_fraction))
    print(comparison_list)

    return comparison_list