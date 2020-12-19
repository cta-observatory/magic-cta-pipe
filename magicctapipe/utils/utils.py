import os
import datetime
import numpy as np
import pandas as pd


def info_message(text, prefix="info"):
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
    """Prints elapsed time as start - end. Output format is `hh:mm:ss`

    Parameters
    ----------
    start : float
        start time, from time.time()
    end : float
        stop time, from time.time()
    """
    h, r = divmod(end - start, 3600)
    m, s = divmod(r, 60)
    print(f"Elapsed time: {int(h):0>2}:{int(m):0>2}:{int(s):0>2}")


def print_title(title, style_char="=", in_space=3, width_char=80):
    """Prints a title string in the following format. If `style_char="="` and 
    `in_space=3`, the string will be:
        ===...=================...===
        ===...===   title   ===...===
        ===...=================...===
    The total width in characters is given by the `width_char` option

    Parameters
    ----------
    title : str
        title string
    style_char : str, optional
        style characted, by default "="
    in_space : int, optional
        number of spaces between the title and the style characters, by default 3
    width_char : int, optional
        total width in characters, by default 80

    Returns
    -------
    str
        formatted title string
    """
    print(
        make_title_str(
            title, style_char=style_char, in_space=in_space, width_char=width_char
        )
    )


def make_title_str(title, style_char="=", in_space=3, width_char=80):
    """Makes a title string in the following format. If `style_char="="` and 
    `in_space=3`, the string will be:
        ===...=================...===
        ===...===   title   ===...===
        ===...=================...===
    The total width in characters is given by the `width_char` option

    Parameters
    ----------
    title : str
        title string
    style_char : str, optional
        style characted, by default "="
    in_space : int, optional
        number of spaces between the title and the style characters, by default 3
    width_char : int, optional
        total width in characters, by default 80

    Returns
    -------
    str
        formatted title string
    """
    s1 = f"{style_char*width_char}\n"
    s2ab_len_float = (width_char - len(title) - in_space * 2) / 2
    if s2ab_len_float > 0:
        s2a_len = int(np.floor(s2ab_len_float))
        s2b_len = int(np.ceil(s2ab_len_float))
        s2a = f"{style_char*s2a_len}{' '*in_space}"
        s2b = f"{' '*in_space}{style_char*s2b_len}\n"
        s2 = f"{s2a}{title}{s2b}"
    else:
        s2 = f"{title}\n"
    s3 = f"{style_char*width_char}"
    s = f"{s1}{s2}{s3}"
    return s
