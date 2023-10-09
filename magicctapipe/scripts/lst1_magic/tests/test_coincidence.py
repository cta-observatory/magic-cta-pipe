import subprocess
import pytest
import glob


def test_coincidence_preoffset(dl1_lst, merge_magic, temp_coinc_preoff, config_preoff):
    """
    Coincidence pre-offset option
    """

    for file in dl1_lst:
        out = subprocess.run(
            [
                "lst1_magic_event_coincidence",
                f"-l{str(file)}",
                f"-m{str(merge_magic)}",
                f"-o{str(temp_coinc_preoff)}",
                f"-c{str(config_preoff)}",
            ]
        )
        if out.returncode != 0:
            raise ValueError(
                f"Coincidence script with pre-offset option failed with return code {out.returncode} for file {file}"
            )    

    assert len(glob.glob(f"{temp_coinc_preoff}/*.h5"))==1


def test_coincidence(dl1_lst, merge_magic, temp_coinc, config):
    """
    Coincidence 
    """

    for file in dl1_lst:
        out = subprocess.run(
            [
                "lst1_magic_event_coincidence",
                f"-l{str(file)}",
                f"-m{str(merge_magic)}",
                f"-o{str(temp_coinc)}",
                f"-c{str(config)}",
            ]
        )
        if out.returncode != 0:
            raise ValueError(
                f"Coincidence script failed with return code {out.returncode} for file {file}"
            )    

    assert len(glob.glob(f"{temp_coinc}/*.h5"))==1




