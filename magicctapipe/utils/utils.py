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
