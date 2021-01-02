import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz
from astropy.coordinates.angle_utilities import position_angle
from astropy.coordinates.angle_utilities import angular_separation
from matplotlib import colors
import matplotlib.pyplot as plt

from magicctapipe.utils.utils import *
from magicctapipe.utils.plot import *
from magicctapipe.utils.tels import *
from magicctapipe.utils.filedir import *


def compute_separation_angle_direction(shower_data_test):
    separation = dict()

    # ???
    shower_data_test.dropna(subset=["az_reco_mean", "alt_reco_mean"], inplace=True)

    tel_ids = get_tel_ids_dl1(shower_data_test)

    for tel_id in tel_ids:
        event_coord_true = SkyCoord(
            shower_data_test.loc[(slice(None), slice(None), tel_id), "true_az"].values
            * u.rad,
            shower_data_test.loc[(slice(None), slice(None), tel_id), "true_alt"].values
            * u.rad,
            frame=AltAz(),
        )

        event_coord_reco = SkyCoord(
            shower_data_test.loc[(slice(None), slice(None), tel_id), "az_reco"].values
            * u.rad,
            shower_data_test.loc[(slice(None), slice(None), tel_id), "alt_reco"].values
            * u.rad,
            frame=AltAz(),
        )

        separation[tel_id] = event_coord_true.separation(event_coord_reco)

    event_coord_true = SkyCoord(
        shower_data_test["true_az"].values * u.rad,
        shower_data_test["true_alt"].values * u.rad,
        frame=AltAz(),
    )

    event_coord_reco = SkyCoord(
        shower_data_test["az_reco_mean"].values * u.rad,
        shower_data_test["alt_reco_mean"].values * u.rad,
        frame=AltAz(),
    )

    separation[0] = event_coord_true.separation(event_coord_reco)

    # Converting to a data frame
    separation_df = pd.DataFrame(
        data={"sep_0": separation[0]}, index=shower_data_test.index
    )
    # ???
    # separation_df = separation_df.dropna()

    for tel_id in tel_ids:
        df = pd.DataFrame(
            data={f"sep_{tel_id:d}": separation[tel_id]},
            index=shower_data_test.loc[
                (slice(None), slice(None), tel_id), "true_az"
            ].index,
        )
        separation_df = separation_df.join(df)

    pd.set_option("display.max_columns", 500)
    print(separation_df)
    separation_df = separation_df.join(shower_data_test)
    print(separation_df)

    for tel_id in [0] + tel_ids:
        s_ = separation[tel_id][~np.isnan(separation[tel_id])]
        print(f"  Tel {tel_id} scatter: ", f"{s_.to(u.deg).std():.2f}")

    return separation_df

