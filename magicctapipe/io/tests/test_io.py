from magicctapipe.io.io import (
    format_object,
    get_dl2_mean,
    get_stereo_events,
    load_lst_dl1_data_file,
    load_magic_dl1_data_files,
    load_train_data_files,
    load_mc_dl2_data_file,
    load_dl2_data_file,
    load_irf_files,
    save_pandas_data_in_table,
)
from magicctapipe.conftest import dl1_file_magic, irf_file
import pytest
import numpy as np
import pandas as pd


def test_format_object():
    """
    Simple check on a string
    """
    str_a = "a{b[[xz,cde}"
    str_b = format_object(str_a)
    assert str_b == "a b  xzcde "


def test_get_stereo_events(stereo_file):
    """
    Check on stereo data reading
    """
    event_data = pd.read_hdf(str(stereo_file), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    data = get_stereo_events(event_data)
    assert np.all(data["multiplicity"] > 1)
    assert np.all(data["combo_type"] >= 0)


def test_get_stereo_events_cut(stereo_file):
    """
    Check on quality cuts
    """
    event_data = pd.read_hdf(str(stereo_file), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    data = get_stereo_events(event_data, "intensity>50")
    assert np.all(data["intensity"] > 50)


def test_get_dl2_mean_mc(dl2_file_mc):
    """
    Check on MC DL2
    """
    event_data = pd.read_hdf(str(dl2_file_mc), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    events = get_dl2_mean(event_data)
    assert "true_energy" in events.columns
    assert events["multiplicity"].dtype == int


def test_get_dl2_mean_real(dl2_file_real):
    """
    Check on real data DL2
    """
    event_data = pd.read_hdf(str(dl2_file_real), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    events = get_dl2_mean(event_data)
    assert "timestamp" in events.columns


def test_get_dl2_mean_avg(dl2_test):
    """
    Check on average evaluation
    """
    event_data = pd.read_hdf(str(dl2_test), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    events = get_dl2_mean(event_data)
    assert np.allclose(np.array(events["gammaness"]), np.array([0.5, 0.6, 1]))


def test_get_dl2_mean_exc(dl2_file_mc):
    """
    Check on exceptions (weight type)
    """
    weight = "abc"
    event_data = pd.read_hdf(str(dl2_file_mc), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    with pytest.raises(ValueError, match=f"Unknown weight type '{weight}'."):
        _ = get_dl2_mean(event_data, weight_type=weight)


def test_load_lst_dl1_data_file(dl1_file_lst):
    """
    Check on LST DL1
    """
    events, _ = load_lst_dl1_data_file(str(dl1_file_lst))
    assert "event_type" in events.columns
    assert "slope" in events.columns
    assert "az_tel" not in events.columns
    events = events.reset_index()
    s = events.duplicated(subset=["obs_id_lst", "event_id_lst"])
    s1 = ~s
    assert s1.all()


def test_load_magic_dl1_data_files(dl1_magic_path, dl1_magic_url, env_prefix):
    """
    Check on MAGIC DL1
    """
    dl1_file_magic(dl1_magic_path, dl1_magic_url, env_prefix)
    events, _ = load_magic_dl1_data_files(str(dl1_magic_path))
    assert list(events.index.names) == ["obs_id_magic", "event_id_magic", "tel_id"]
    assert "event_id" not in events.columns
    events = events.reset_index()
    s = events.duplicated(subset=["obs_id_magic", "event_id_magic", "tel_id"])
    s1 = ~s
    assert s1.all()


def test_load_magic_dl1_data_files_exc(dl1_magic_exc_path):
    """
    Check on MAGIC DL1: exceptions (no DL1 files)
    """
    with pytest.raises(
        FileNotFoundError,
        match="Could not find any DL1 data files in the input directory.",
    ):
        _, _ = load_magic_dl1_data_files(str(dl1_magic_exc_path))


def test_load_train_data_files(stereo_path):
    """
    Check dictionary
    """
    events = load_train_data_files(str(stereo_path))
    assert list(events.keys()) == ["M1_M2", "LST1_M1", "LST1_M2", "LST1_M1_M2"]
    data = events["LST1_M1"]
    assert np.all(data["combo_type"]) == 1
    assert "off_axis" in data.columns
    assert "true_event_class" not in data.columns


def test_load_train_data_files_off(stereo_path):
    """
    Check off-axis cut
    """
    events = load_train_data_files(
        str(stereo_path), offaxis_min="0.2 deg", offaxis_max="0.5 deg"
    )
    data = events["LST1_M1"]
    assert np.all(data["off_axis"] >= 0.2)
    assert np.all(data["off_axis"] <= 0.5)


def test_load_train_data_files_exc(stereo_path_exc):
    """
    Check on exceptions
    """
    with pytest.raises(
        FileNotFoundError,
        match="Could not find any DL1-stereo data files in the input directory.",
    ):
        _ = load_train_data_files(str(stereo_path_exc))


def test_load_mc_dl2_data_file(dl2_file_mc):
    """
    Checks on default loading
    """
    data, point, _ = load_mc_dl2_data_file(
        str(dl2_file_mc), "width>0", "software", "simple"
    )
    assert "pointing_alt" in data.colnames
    assert "theta" in data.colnames
    assert "true_source_fov_offset" in data.colnames
    assert data["true_energy"].unit == "TeV"
    assert point[0] >= 0
    assert point[0] <= 90


def test_load_mc_dl2_data_file_cut(dl2_file_mc):
    """
    Check on quality cuts
    """
    data, _, _ = load_mc_dl2_data_file(
        str(dl2_file_mc), "gammaness>0.9", "software", "simple"
    )
    assert np.all(data["gammaness"] > 0.9)


def test_load_mc_dl2_data_file_opt(dl2_file_mc):
    """
    Check on event_type
    """
    data_s, _, _ = load_mc_dl2_data_file(
        str(dl2_file_mc), "width>0", "software", "simple"
    )
    data_m, _, _ = load_mc_dl2_data_file(
        str(dl2_file_mc), "width>0", "magic_only", "simple"
    )
    assert np.all(data_s["combo_type"] > 0)
    assert np.all(data_m["combo_type"] == 0)


def test_load_mc_dl2_data_file_exc(dl2_file_mc):
    """
    Check on event_type exceptions
    """
    event_type = "abc"
    with pytest.raises(
        ValueError,
        match=f"Unknown event type '{event_type}'.",
    ):
        _, _, _ = load_mc_dl2_data_file(
            str(dl2_file_mc), "width>0", event_type, "simple"
        )


def test_load_dl2_data_file(dl2_file_real):
    """
    Checks on default loading
    """
    data, on, dead = load_dl2_data_file(
        str(dl2_file_real), "width>0", "software", "simple"
    )
    assert "pointing_alt" in data.colnames
    assert "timestamp" in data.colnames
    assert data["reco_energy"].unit == "TeV"
    assert on.unit == "s"
    assert on > 0
    assert dead > 0


def test_load_dl2_data_file_cut(dl2_file_real):
    """
    Check on quality cuts
    """
    data, _, _ = load_dl2_data_file(
        str(dl2_file_real), "gammaness>0.9", "software", "simple"
    )
    assert np.all(data["gammaness"] > 0.9)


def test_load_dl2_data_file_opt(dl2_file_real):
    """
    Check on event_type
    """
    data_s, _, _ = load_dl2_data_file(
        str(dl2_file_real), "width>0", "software", "simple"
    )
    data_m, _, _ = load_dl2_data_file(
        str(dl2_file_real), "width>0", "magic_only", "simple"
    )
    assert np.all(data_s["combo_type"] > 0)
    assert np.all(data_m["combo_type"] == 0)


""" def test_load_dl2_data_file_exc(dl2_file_real):
    
    Check on event_type exceptions
    
    event_type="abc"
    with pytest.raises(
        ValueError,
        match=f"Unknown event type '{event_type}'.",
    ):
        _,_,_=load_dl2_data_file(str(dl2_file_real),"width>0",event_type,"simple")

 """


def test_load_irf_files(irf_path, irf_url, env_prefix):
    """
    Check on IRF dictionaries
    """
    irf_file(irf_path, irf_url, env_prefix)
    irf, header = load_irf_files(str(irf_path))
    assert set(list(irf.keys())).issubset(
        set(
            [
                "grid_points",
                "effective_area",
                "energy_dispersion",
                "psf_table",
                "background",
                "gh_cuts",
                "rad_max",
                "energy_bins",
                "fov_offset_bins",
                "migration_bins",
                "source_offset_bins",
                "bkg_fov_offset_bins",
            ]
        )
    )
    assert len(irf["effective_area"][0][0]) > 0
    assert "psf_table" not in list(irf.keys())
    assert "background" not in list(irf.keys())
    assert set(list(header.keys())).issubset(
        set(
            [
                "TELESCOP",
                "INSTRUME",
                "FOVALIGN",
                "QUAL_CUT",
                "EVT_TYPE",
                "DL2_WEIG",
                "IRF_OBST",
                "GH_CUT",
                "GH_EFF",
                "GH_MIN",
                "GH_MAX",
                "RAD_MAX",
                "TH_EFF",
                "TH_MIN",
                "TH_MAX",
            ]
        )
    )
    assert header["DL2_WEIG"] == "simple"
    assert header["EVT_TYPE"] == "software"


def test_load_irf_files_exc(irf_path_exc):
    """
    Check on exception (FileNotFound)
    """
    with pytest.raises(
        FileNotFoundError,
        match="Could not find any IRF data files in the input directory.",
    ):
        _, _ = load_irf_files(str(irf_path_exc))


def test_save_pandas_data_in_table(pd_path, pd_test):
    """
    Check on pandas dataframe (before=after saving it)
    """
    out = pd_path / "pandas.h5"
    save_pandas_data_in_table(pd_test, str(out), "abc", "event")
    df1 = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    df = pd.read_hdf(str(out), key="event")
    assert df.equals(df1)
