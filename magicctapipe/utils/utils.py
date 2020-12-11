import os
import datetime
import pandas as pd


def info_message(text, prefix='info'):
    """Prints the specified text with the prefix of the current date

    Parameters
    ----------
    text : str
        text
    prefix : str, optional
        prefix, by default 'info'
    """
    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"({prefix:s}) {date_str:s}: {text:s}")


def print_elapsed_time(start, end):
    """Print elapsed time as start - end. Output format is `hh:mm:ss`

    Parameters
    ----------
    start : float
        start time, from time.time()
    end : float
        stop time, from time.time()
    """
    h, r = divmod(end-start, 3600)
    m, s = divmod(r, 60)
    print(f"Elapsed time: {int(h):0>2}:{int(m):0>2}:{int(s):0>2}")
