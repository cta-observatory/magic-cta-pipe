import numpy as np


def add_delta_t_key(events):
    """
    Adds the time difference with the previous event to a real data
    dataframe.
    Should be only used only with non-filtered data frames,
    so events are consecutive.
    Parameters
    ----------
    events: pandas DataFrame of dl1 events
    Returns
    -------
    events: pandas DataFrame of dl1 events with delta_t
    """

    # Get delta t of real data and add it to the data frame
    if "dragon_time" in events.columns:
        time = np.array(events["dragon_time"])
        delta_t = np.insert(np.diff(time), 0, 0)
        events["delta_t"] = delta_t
    return events
