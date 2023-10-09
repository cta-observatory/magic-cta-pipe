import subprocess
import pytest
import glob


def test_mc_dl0_dl1(temp_DL1_gamma, dl0_gamma, config):
    """
    MC DL0 to DL1
    """

    for file in dl0_gamma:
        out=subprocess.run(
            [
                "lst1_magic_mc_dl0_to_dl1",
                f"-i{str(file)}",
                f"-o{str(temp_DL1_gamma)}",
                f"-c{str(config)}",
            ]
        )
    
        if out.returncode != 0:
            raise ValueError(
                f"MC DL0 to DL1 script failed with return code {out.returncode} for file {file}"
            )    

    assert len(glob.glob(f"{temp_DL1_gamma}/*.h5"))>0


def test_mc_dl0_dl1_focal_exc(temp_DL1_gamma_focal_exc, dl0_gamma, config):
    """
    MC DL0 to DL1 focal length exception
    """

    for file in dl0_gamma:
        out=subprocess.run(
            [
                "lst1_magic_mc_dl0_to_dl1",
                f"-i{str(file)}",
                f"-o{str(temp_DL1_gamma_focal_exc)}",
                f"-c{str(config)}",
                "-f abc",
            ]
        )
    assert len(glob.glob(f"{temp_DL1_gamma_focal_exc}/*.h5"))==0
    
        
