import os


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
