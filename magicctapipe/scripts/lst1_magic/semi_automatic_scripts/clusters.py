"""
Module for generating bash script lines for running analysis in different clusters
"""


def slurm_lines(p, J, array=None, mem=None, out_err=None):
    """
    Function for creating the general lines that slurm scripts are starting with.

    Parameters
    ----------
    p : str
        Name of the queue
    J : str
        Job name
    array : None or int
        If not none array of jobs from 0 to array will be made
    mem : None or str
        Requested memory
    out_err : None or str
        If the output should be written to a specific output file

    Returns
    -------
    list
        List of strings
    """
    lines = [
        "#!/bin/sh\n\n",
        f"#SBATCH -p {p}\n",
        f"#SBATCH -J {J}\n",
        f"#SBATCH --array=0-{array}\n" if array is not None else "",
        f"#SBATCH --mem {mem}\n" if mem is not None else "",
        "#SBATCH -n 1\n\n",
        f"#SBATCH --output={out_err}.out\n" if out_err is not None else "",
        f"#SBATCH --error={out_err}.err\n\n" if out_err is not None else "",
        "ulimit -l unlimited\n",
        "ulimit -s unlimited\n",
        "ulimit -a\n\n",
    ]
    return lines
