import os
import datetime


def check_folder(folder):
    """Check if folder exists; if not, it will be created"""
    if not os.path.exists(folder):
        print("Directory %s does not exist, creating it..." % folder)
        os.makedirs(folder)

def out_file_h5(in_file, li, hi):
    """Returns the h5 output file name, from a simtel.gz input file

    Parameters
    ----------
    in_file : str
        Input file
    li : int
        low index
    hi : int
        high index

    Returns
    -------
    str
        h5 output file, absolute path
    """
    f = os.path.basename(in_file)
    out = '_'.join(f.split('_')[:li]+f.split('_')[hi:])
    out = '%s.h5' % out.rstrip('.simtel.gz')
    out = os.path.join(os.path.dirname(in_file), out)
    return out


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
