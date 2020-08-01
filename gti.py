# coding: utf-8

import glob
import yaml
import argparse
import datetime

import scipy
import uproot
import pandas


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


def identify_time_edges(times, criterion, max_time_diff=6.9e-4):
    """
    Identifies the time interval edges, corresponding to the True
    state of the specified condition. Neighbouring time intervals, 
    separated by not more than max_time_diff are joined together.

    Parameters
    ----------
    times: array_like
        Array of the time data points.
    criterion: array_like
        Array of True/False values, indicating the goodness of the 
        corresponding data points.
    max_time_diff: float, optional
        Maximal time difference between the time intervals, below which 
        they are joined into one.

    Returns
    -------
    parts_edges: list
        List of start/stop pairs, describing the identified time intervals.

    """

    times = scipy.array(times)
    wh = scipy.where(criterion == True)
    
    if len(wh[0]) == 0:
        print("No time intervals excluded!")
        
        return [[0,0]]
    
    # The above function will result in indicies of non-zero bins.
    # But we want indicies of their _edges_.
    non_zero_bin_edges = []
    
    for i in wh[0]:
        if i-1 >= 0:
            if abs(times[i]-times[i-1]) < max_time_diff:
                non_zero_bin_edges.append(i-1)
        
        non_zero_bin_edges.append(i)
        
        if i+1 < len(times):
            if abs(times[i]-times[i+1]) < max_time_diff:
                non_zero_bin_edges.append(i+1)
        
    non_zero_bin_edges = scipy.unique(non_zero_bin_edges)
    
    if len(non_zero_bin_edges) > 2:
        # Finding the elements, that separate the observational time intervals
        # During one time interval diff should return 1 (current bin has index i, the index of the next time bin is i+1).
        # Here we're looking for diffs that are not equal to 1.
        #division_indicies = (scipy.diff(non_zero_bin_edges[1:-1])-1).nonzero()
        #division_indicies = division_indicies[0]
        cond_id = scipy.diff(non_zero_bin_edges[1:-1]) > 1
        cond_time = scipy.diff(times[non_zero_bin_edges[1:-1]]) > max_time_diff
        division_indicies = scipy.where(cond_id | cond_time)
        division_indicies = division_indicies[0]
        
        # Concatenating to the found elements the beginning and the end of the observational time.
        # Also adding i+1 elements, to correctly switch to the next observational time interval.
        parts_edges_idx = scipy.concatenate(
                (
                    [non_zero_bin_edges[0]], 
                    non_zero_bin_edges[1:-1][division_indicies], 
                    non_zero_bin_edges[1:-1][division_indicies+1], 
                    [non_zero_bin_edges[-1]]
                )
            )
    else:
        parts_edges_idx = scipy.array(non_zero_bin_edges)
    parts_edges_idx.sort()
    
    # Transorming edges indicies to the real values and transforming them to the [start, stop] list.
    parts_edges = times[parts_edges_idx]
    parts_edges = parts_edges.reshape(-1,2)
    
    return parts_edges


def intersect_time_intervals(intervals1, intervals2):
    """
    Intersects two lists of (TStart, TStop) pairs. Returned list 
    contains the start/stop invervals, common in both input lists.

    Parameters
    ----------
    interval1: list
        First list of (TStart, TStop) lists (or tuples).
    interval2: list
        Second list of (TStart, TStop) lists (or tuples).

    Returns
    -------
    joint_intervals: list
        A list of (TStart, TStop) lists, representing the start/stop invervals, 
        common in both input lists.

    """


    joined_intervals  = []
    
    # Comparing 1st to 2nd
    for interv1 in intervals1:
        tstart = None
        tstop = None
        
        for interv2 in intervals2:
            if (interv2[0] >= interv1[0]) and (interv2[0] <= interv1[1]):
                tstart = interv2[0]
                tstop = min(interv2[1], interv1[1])
            elif (interv2[1] >= interv1[0]) and (interv2[1] <= interv1[1]):
                tstart = max(interv2[0], interv1[0])
                tstop = interv2[1]
            elif (interv2[0] < interv1[0]) and (interv2[1] > interv1[1]):
                tstart = interv1[0]
                tstop = interv1[1]

        if tstart and tstop:
            joined_intervals.append([tstart, tstop])
 
    return joined_intervals

class GTIGenerator:
    def __init__(self, config=None, verbose=False):
        self._config = config
        self.verbose = verbose

    @property
    def config(self):
        return self._config.copy()

    @config.setter
    def config(self, new_config):
        if 'event_list' not in new_config:
            raise ValueError('GTI generator error: the configuration dict is missing the "event_list" section.')
        
        self._config = new_config.copy()

    def _itentify_data_taking_time_edges(self, file_list, max_tdiff=1):
        if not file_list:
            raise ValueError("GTI generator: no files to process")

        mjd = scipy.zeros(0)
        tdiff = scipy.zeros(0)

        if self.verbose:
            info_message(f"identifying data taking time edges", "GTI generator")

        for fnum, fname in enumerate(file_list):
            with uproot.open(fname) as input_stream:
                if self.verbose:
                    info_message(f"processing file {fnum+1} / {len(file_list)}", "GTI generator")

                _tdiff = input_stream['Events']['MRawEvtHeader.fTimeDiff'].array()
                
                _mjd = input_stream['Events']['MTime.fMjd'].array()
                _millisec = input_stream['Events']['MTime.fTime.fMilliSec'].array()
                _nanosec = input_stream['Events']['MTime.fNanoSec'].array()

                _mjd = _mjd + (_millisec / 1e3 + _nanosec / 1e9) / 86400.0
                
                mjd = scipy.concatenate((mjd, _mjd))
                tdiff = scipy.concatenate((tdiff, _tdiff))

        sort_args = mjd.argsort()

        not_edge = tdiff < max_tdiff

        time_intervals = identify_time_edges(
            mjd[sort_args], 
            not_edge[sort_args], 
            max_time_diff=max_tdiff / 86400.0
        )

        return time_intervals

    def _itentify_dc_time_edges(self, file_list):
        if not file_list:
            raise ValueError("GTI generator: no files to process")

        if "dc" not in self.config['event_list']['cuts']['quality']:
            raise ValueError("GTI generator: no DC cuts given")

        if self.verbose:
            info_message(f"identifying DC time edges", "GTI generator")

        df = pandas.DataFrame()

        # Looping over the data files
        for fnum, file_name in enumerate(file_list):
            if self.verbose:
                info_message(f"processing file {fnum+1} / {len(file_list)}", "GTI generator")

            with uproot.open(file_name) as input_stream:
                mjd = input_stream["Camera"]["MTimeCamera.fMjd"].array()
                millisec = input_stream["Camera"]["MTimeCamera.fTime.fMilliSec"].array()
                nanosec = input_stream["Camera"]["MTimeCamera.fNanoSec"].array()

                df_ = pandas.DataFrame()
                    
                df_['mjd'] = mjd + (millisec / 1e3 + nanosec / 1e9) / 86400
                df_['value'] = input_stream["Camera"]["MReportCamera.fMedianDC"].array()
                    
                df = df.append(df_)

        df = df.sort_values(by=['mjd'])

        cut = self.config['event_list']['cuts']['quality']['dc']
           
        selection = df.query(cut)

        _, idx, _ = scipy.intersect1d(df["mjd"], selection["mjd"], return_indices=True)

        criterion = scipy.repeat(False, len(df["mjd"]))
        criterion[idx] = True

        time_intervals = identify_time_edges(
            df["mjd"], 
            criterion, 
            max_time_diff=self.config['event_list']['max_time_diff']
        )

        return time_intervals

    def _itentify_l3rate_time_edges(self, file_list):
        if not file_list:
            raise ValueError("GTI generator: no files to process")

        if "l3rate" not in self.config['event_list']['cuts']['quality']:
            raise ValueError("GTI generator: no L3 rate cuts given")

        if self.verbose:
            info_message(f"identifying L3 rate time edges", "GTI generator")

        df = pandas.DataFrame()

        # Looping over the data files
        for fnum, file_name in enumerate(file_list):
            info_message(f"processing file {fnum+1} / {len(file_list)}", "GTI generator")

            with uproot.open(file_name) as input_stream:
                mjd = input_stream["Trigger"]["MTimeTrigger.fMjd"].array()
                millisec = input_stream["Trigger"]["MTimeTrigger.fTime.fMilliSec"].array()
                nanosec = input_stream["Trigger"]["MTimeTrigger.fNanoSec"].array()

                df_ = pandas.DataFrame()
                    
                df_['mjd'] = mjd + (millisec / 1e3 + nanosec / 1e9) / 86400
                df_['value'] = input_stream["Trigger"]["MReportTrigger.fL3Rate"].array()
                    
                    
                df = df.append(df_)

        df = df.sort_values(by=['mjd'])

        cut = self.config['event_list']['cuts']['quality']['l3rate']
           
        selection = df.query(cut)

        _, idx, _ = scipy.intersect1d(df["mjd"], selection["mjd"], return_indices=True)

        criterion = scipy.repeat(False, len(df["mjd"]))
        criterion[idx] = True

        time_intervals = identify_time_edges(
            df["mjd"], 
            criterion, 
            max_time_diff=self.config['event_list']['max_time_diff']
        )

        return time_intervals
        

    def process_files(self, file_list):
        """
        GTI list generator.

        Parameters
        ----------
        config: dict
            Configuration dictionary (e.g. read from a YAML file).

        Returns
        -------
        joint_intervals: list
            A list of (TStart, TStop) lists, representing the identified GTIs.

        """

        if not self.config:
            raise ValueError("GTIGenerator: configuration is not set")

        # # Identifying the files to read
        # info_message("looking for the files", "GTI generator")
        # file_list = glob.glob(file_mask)

        # if not file_list:
        #     raise ValueError("No files to process")

        # # Containers for the data points
        # dfs = {
        #     'dc': pandas.DataFrame(),
        #     'l3rate': pandas.DataFrame()
        # }

        # # Removing the containers, not specified in the configuration card
        # if "l3rate" not in config['event_list']['cuts']:
        #     del dfs['l3rate']

        # if "dc" not in config['event_list']['cuts']:
        #     del dfs['dc']

        # # Looping over the data files
        # for fnum, file_name in enumerate(file_list):
        #     info_message(f"processing file {fnum+1} / {len(file_list)}", "GTI generator")

        #     with uproot.open(file_name) as input_stream:
                
        #         # --- DC ---
        #         if "dc" in dfs:
        #             mjd = input_stream["Camera"]["MTimeCamera.fMjd"].array()
        #             millisec = input_stream["Camera"]["MTimeCamera.fTime.fMilliSec"].array()
        #             nanosec = input_stream["Camera"]["MTimeCamera.fNanoSec"].array()

        #             df_ = pandas.DataFrame()
                    
        #             df_['mjd'] = mjd + (millisec / 1e3 + nanosec / 1e9) / 86400
        #             df_['value'] = input_stream["Camera"]["MReportCamera.fMedianDC"].array()
                    
        #             dfs['dc'] = dfs['dc'].append(df_)
                
        #         # --- L3 rate ---
        #         if "l3rate" in dfs:
        #             mjd = input_stream["Trigger"]["MTimeTrigger.fMjd"].array()
        #             millisec = input_stream["Trigger"]["MTimeTrigger.fTime.fMilliSec"].array()
        #             nanosec = input_stream["Trigger"]["MTimeTrigger.fNanoSec"].array()

        #             df_ = pandas.DataFrame()
                    
        #             df_['mjd'] = mjd + (millisec / 1e3 + nanosec / 1e9) / 86400
        #             df_['value'] = input_stream["Trigger"]["MReportTrigger.fL3Rate"].array()
                    
                    
        #             dfs['l3rate'] = dfs['l3rate'].append(df_)

        # # Sorting data points by date is needed for the time intervals identification
        # for key in dfs:
        #     dfs[key] = dfs[key].sort_values(by=['mjd'])

        # info_message("identifying GTIs", "GTI generator")

        # time_intervals_list = []

        # # Identifying DC-related GTIs
        # if "dc" in dfs:
        #     cut = config['event_list']['cuts']['dc']
            
        #     selection = dfs['dc'].query(cut)

        #     _, idx, _ = scipy.intersect1d(dfs['dc']["mjd"], selection["mjd"], return_indices=True)

        #     criterion = scipy.repeat(False, len(dfs['dc']["mjd"]))
        #     criterion[idx] = True

        #     time_intervals = identify_time_edges(dfs['dc']["mjd"], criterion, max_time_diff=config['event_list']['max_time_diff'])

        #     time_intervals_list.append(time_intervals)

        # # Identifying L3-related GTIs
        # if "l3rate" in dfs:
        #     cut = config['event_list']['cuts']['l3rate']
            
        #     selection = dfs['l3rate'].query(cut)

        #     _, idx, _ = scipy.intersect1d(dfs['l3rate']["mjd"], selection["mjd"], return_indices=True)

        #     criterion = scipy.repeat(False, len(dfs['l3rate']["mjd"]))
        #     criterion[idx] = True

        #     time_intervals = identify_time_edges(dfs['l3rate']["mjd"], criterion, max_time_diff=config['event_list']['max_time_diff'])

        #     time_intervals_list.append(time_intervals)

        time_intervals_list = [
            self._itentify_data_taking_time_edges(file_list),
            self._itentify_dc_time_edges(file_list),
            self._itentify_l3rate_time_edges(file_list)
        ]

        joint_intervals = time_intervals_list[0]

        # Joining all found GTIs
        for i in range(1, len(time_intervals_list)):
            joint_intervals = intersect_time_intervals(joint_intervals, time_intervals_list[i])

        return joint_intervals
    

# =================
# === Main code ===
# =================

if __name__ == "__main__":
    # --------------------------
    # Adding the argument parser
    arg_parser = argparse.ArgumentParser(description="""
    This tools fits the direction random forest regressor on the specified events files.
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

    if 'event_list' not in config:
        print('Error: the configuration file is missing the "event_list" section. Exiting.')
        exit()
    # ------------------------------

    joint_intervals = gti_generator(config)

    print("--- Identified GTI intervals [MJD] ---")
    for interval in joint_intervals:
        print(interval)