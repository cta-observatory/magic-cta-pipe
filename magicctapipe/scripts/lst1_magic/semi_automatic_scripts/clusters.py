"""
Module for generating bash script lines for running analysis in different clusters
"""
__all__ = ["slurm_lines", "rc_lines"]


def slurm_lines(
    queue, job_name, nice_parameter=None, array=None, mem=None, out_name=None
):

    """
    Function for creating the general lines that slurm scripts are starting with.

    Parameters
    ----------
    queue : str
        Name of the queue
    job_name : str
        Job name
    nice_parameter : int or None
        Job priority
    array : None or int
        If not none array of jobs from 0 to array will be made
    mem : None or str
        Requested memory. If None cluster default (5 GB) will be used
    out_name : None or str
        If the output should be written to a specific output file

    Returns
    -------
    list
        List of strings to submit a SLURM job.
    """
    lines = [
        "#!/bin/sh\n\n",
        f"#SBATCH -p {queue}\n",
        f"#SBATCH -J {job_name}\n",
        f"#SBATCH --array=0-{array}\n" if array is not None else "",
        f"#SBATCH --mem {mem}\n" if mem is not None else "",
        "#SBATCH -n 1\n\n",
        f"#SBATCH --output={out_name}.out\n" if out_name is not None else "",
        f"#SBATCH --error={out_name}.err\n\n" if out_name is not None else "",
        f"#SBATCH --nice={nice_parameter}\n" if nice_parameter is not None else "",
        "ulimit -l unlimited\n",
        "ulimit -s unlimited\n",
        "ulimit -a\n\n",
    ]
    return lines


def rc_lines(store, out):
    """
    Function for creating the general lines for error tracking.

    Parameters
    ----------
    store : str
        String what to store in addition to $rc
    out : str
        Base name for the log files with return codes, all output will go into {out}_return.log, only errors to {out}_failed.log

    Returns
    -------
    list
        List of strings to attach to a shell script
    """
    lines = [
        "rc=$?\n",
        'if [ "$rc" -ne "0" ]; then\n',
        f"  echo {store} $rc >> {out}_failed.log\n",
        "fi\n",
        f"echo {store} $rc >> {out}_return.log\n",
    ]
    return lines
