import numpy as np
import pandas as pd
import pytest

from magicctapipe.io.io import (
    check_input_list,
    format_object,
    get_dl2_mean,
    get_stereo_events,
    load_dl2_data_file,
    load_irf_files,
    load_lst_dl1_data_file,
    load_magic_dl1_data_files,
    load_mc_dl2_data_file,
    load_train_data_files,
    load_train_data_files_tel,
    save_pandas_data_in_table,
    telescope_combinations,
)


def test_check_input_list(config_check):
    """
    Test on different dictionaries
    """

    check_input_list(
        {
            "mc_tel_ids": {
                "LST-1": 1,
                "LST-2": 2,
                "LST-3": 3,
                "LST-4": 4,
                "MAGIC-I": 5,
                "MAGIC-II": 6,
            }
        }
    )

    check_input_list(
        {
            "mc_tel_ids": {
                "LST-1": 1,
                "LST-2": 3,
                "LST-3": 0,
                "LST-4": 0,
                "MAGIC-I": 2,
                "MAGIC-II": 6,
            }
        }
    )

    check_input_list(
        {
            "mc_tel_ids": {
                "LST-2": 1,
                "LST-1": 3,
                "LST-4": 0,
                "LST-3": 0,
                "MAGIC-II": 2,
                "MAGIC-I": 6,
            }
        }
    )

    with pytest.raises(
        Exception,
        match="Number of telescopes found in the configuration file is 5. It must be 6, i.e.: LST-1, LST-2, LST-3, LST-4, MAGIC-I, and MAGIC-II.",
    ):
        check_input_list(config_check)

    with pytest.raises(
        Exception,
        match="Number of telescopes found in the configuration file is 5. It must be 6, i.e.: LST-1, LST-2, LST-3, LST-4, MAGIC-I, and MAGIC-II.",
    ):
        check_input_list(
            {
                "mc_tel_ids": {
                    "LST-1": 1,
                    "LST-2": 2,
                    "LST-3": 3,
                    "MAGIC-I": 4,
                    "MAGIC-II": 5,
                }
            }
        )

    with pytest.raises(
        Exception,
        match="Number of telescopes found in the configuration file is 7. It must be 6, i.e.: LST-1, LST-2, LST-3, LST-4, MAGIC-I, and MAGIC-II.",
    ):
        check_input_list(
            {
                "mc_tel_ids": {
                    "LST-1": 1,
                    "LST-2": 2,
                    "LST-3": 3,
                    "LST-4": 6,
                    "LST-5": 7,
                    "MAGIC-I": 4,
                    "MAGIC-II": 5,
                }
            }
        )

    with pytest.raises(
        Exception,
        match="Entry 'LSTT-1' not accepted as an LST. Please make sure that the first four telescopes are LSTs, e.g.: 'LST-1', 'LST-2', 'LST-3', and 'LST-4'",
    ):
        check_input_list(
            {
                "mc_tel_ids": {
                    "LSTT-1": 1,
                    "LST-2": 2,
                    "LST-3": 3,
                    "LST-4": 6,
                    "MAGIC-I": 4,
                    "MAGIC-II": 5,
                }
            }
        )

    with pytest.raises(
        Exception,
        match="Entry 'MAGIC-III' not accepted as a MAGIC. Please make sure that the last two telescopes are MAGICs, e.g.: 'MAGIC-I', and 'MAGIC-II'",
    ):
        check_input_list(
            {
                "mc_tel_ids": {
                    "LST-1": 1,
                    "LST-2": 2,
                    "LST-3": 3,
                    "LST-4": 6,
                    "MAGIC-I": 4,
                    "MAGIC-III": 5,
                }
            }
        )

    with pytest.raises(
        Exception,
        match="Entry 'MAGIC-I' not accepted as an LST. Please make sure that the first four telescopes are LSTs, e.g.: 'LST-1', 'LST-2', 'LST-3', and 'LST-4'",
    ):
        check_input_list(
            {
                "mc_tel_ids": {
                    "LST-1": 1,
                    "LST-2": 2,
                    "MAGIC-I": 4,
                    "LST-3": 3,
                    "LST-4": 6,
                    "MAGIC-II": 5,
                }
            }
        )


def test_telescope_combinations(config_gen, config_gen_4lst):
    """
    Simple check on telescope combinations
    """
    M_LST, M_LST_comb = telescope_combinations(config_gen)
    LSTs, LSTs_comb = telescope_combinations(config_gen_4lst)
    assert M_LST == {1: "LST-1", 2: "MAGIC-I", 3: "MAGIC-II"}
    assert M_LST_comb == {
        "M1_M2": [2, 3],  # combo_type = 0
        "LST1_M1": [1, 2],  # combo_type = 1
        "LST1_M2": [1, 3],  # combo_type = 2
        "LST1_M1_M2": [1, 2, 3],  # combo_type = 3
    }  # TODO: change in next PR
    assert list(M_LST_comb.keys()) == [
        "M1_M2",
        "LST1_M1",
        "LST1_M2",
        "LST1_M1_M2",
    ]  # TODO: change in next PR
    assert LSTs == {1: "LST-1", 3: "LST-2", 2: "LST-3", 5: "LST-4"}
    assert LSTs_comb == {
        "LST-1_LST-2": [1, 3],
        "LST-1_LST-2_LST-3": [1, 3, 2],
        "LST-1_LST-2_LST-3_LST-4": [1, 3, 2, 5],
        "LST-1_LST-2_LST-4": [1, 3, 5],
        "LST-1_LST-3": [1, 2],
        "LST-1_LST-3_LST-4": [1, 2, 5],
        "LST-1_LST-4": [1, 5],
        "LST-2_LST-3": [3, 2],
        "LST-2_LST-3_LST-4": [3, 2, 5],
        "LST-2_LST-4": [3, 5],
        "LST-3_LST-4": [2, 5],
    }


def test_format_object():
    """
    Simple check on a string
    """
    str_a = "a{b[[xz,cde}"
    str_b = format_object(str_a)
    assert str_b == "a b  xzcde "


def test_save_pandas_data_in_table(temp_pandas, pd_test):
    """
    Check on pandas dataframe (before = after saving it)
    """
    out = temp_pandas / "pandas.h5"
    save_pandas_data_in_table(pd_test, str(out), "abc", "event")
    df1 = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
    df = pd.read_hdf(str(out), key="event")
    assert df.equals(df1)


def test_get_stereo_events_mc(gamma_stereo, p_stereo, config_gen):
    """
    Check on stereo data reading
    """

    stereo_mc = (
        [p for p in gamma_stereo[0].glob("*")]
        + [p for p in gamma_stereo[1].glob("*")]
        + [p for p in p_stereo[0].glob("*")]
        + [p for p in p_stereo[1].glob("*")]
    )

    for file in stereo_mc:
        event_data = pd.read_hdf(str(file), key="events/parameters")
        event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
        event_data.sort_index(inplace=True)
        data = get_stereo_events(event_data, config_gen)
        assert np.all(data["multiplicity"] > 1)
        assert np.all(data["combo_type"] >= 0)


def test_get_stereo_events_mc_cut(gamma_stereo, p_stereo, config_gen):
    """
    Check on quality cuts
    """
    stereo_mc = (
        [p for p in gamma_stereo[0].glob("*")]
        + [p for p in gamma_stereo[1].glob("*")]
        + [p for p in p_stereo[0].glob("*")]
        + [p for p in p_stereo[1].glob("*")]
    )
    for file in stereo_mc:
        event_data = pd.read_hdf(str(file), key="events/parameters")
        event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
        event_data.sort_index(inplace=True)
        data = get_stereo_events(event_data, config_gen, "intensity>50")
        assert np.all(data["intensity"] > 50)


def test_load_train_data_files_p(p_stereo):
    """
    Check dictionary
    """

    events = load_train_data_files(str(p_stereo[0]))
    assert list(events.keys()) == [
        "M1_M2",
        "LST1_M1",
        "LST1_M2",
        "LST1_M1_M2",
    ]  # TODO: change in next PR
    data = events["LST1_M1"]
    assert np.all(data["combo_type"] == 1)  # TODO: change in next PR
    assert "off_axis" in data.columns
    assert "true_event_class" not in data.columns


def test_load_train_data_files_g(gamma_stereo):
    """
    Check dictionary
    """

    events = load_train_data_files(str(gamma_stereo[0]))
    assert list(events.keys()) == [
        "M1_M2",
        "LST1_M1",
        "LST1_M2",
        "LST1_M1_M2",
    ]  # TODO: change in next PR
    data = events["LST1_M1"]
    assert np.all(data["combo_type"] == 1)  # TODO: change in next PR
    assert "off_axis" in data.columns
    assert "true_event_class" not in data.columns


def test_load_train_data_files_off(gamma_stereo):
    """
    Check off-axis cut
    """
    events = load_train_data_files(
        str(gamma_stereo[0]), offaxis_min="0.2 deg", offaxis_max="0.5 deg"
    )
    data = events["LST1_M1"]
    assert np.all(data["off_axis"] >= 0.2)
    assert np.all(data["off_axis"] <= 0.5)


def test_load_train_data_files_exc(temp_train_exc):
    """
    Check on exceptions
    """
    with pytest.raises(
        FileNotFoundError,
        match="Could not find any DL1-stereo data files in the input directory.",
    ):
        _ = load_train_data_files(str(temp_train_exc))


def test_load_train_data_files_tel_p(p_stereo, config_gen):
    """
    Check dictionary
    """

    events = load_train_data_files_tel(str(p_stereo[0]), config_gen)
    assert list(events.keys()) == [1, 2, 3]
    data = events[2]
    assert "off_axis" in data.columns
    assert "true_event_class" not in data.columns


def test_load_train_data_files_tel_g(gamma_stereo, config_gen):
    """
    Check dictionary
    """

    events = load_train_data_files_tel(str(gamma_stereo[0]), config_gen)
    assert list(events.keys()) == [1, 2, 3]
    data = events[1]
    assert "off_axis" in data.columns
    assert "true_event_class" not in data.columns


def test_load_train_data_files_tel_off(gamma_stereo, config_gen):
    """
    Check off-axis cut
    """
    events = load_train_data_files_tel(
        str(gamma_stereo[0]),
        config=config_gen,
        offaxis_min="0.2 deg",
        offaxis_max="0.5 deg",
    )
    data = events[1]
    assert np.all(data["off_axis"] >= 0.2)
    assert np.all(data["off_axis"] <= 0.5)


def test_load_train_data_files_tel_exc(temp_train_exc, config_gen):
    """
    Check on exceptions
    """
    with pytest.raises(
        FileNotFoundError,
        match="Could not find any DL1-stereo data files in the input directory.",
    ):
        _ = load_train_data_files(str(temp_train_exc), config_gen)


def test_load_mc_dl2_data_file(p_dl2, gamma_dl2):
    """
    Checks on default loading
    """
    dl2_mc = [p for p in gamma_dl2.glob("*")] + [p for p in p_dl2.glob("*")]
    for file in dl2_mc:
        data, point, _ = load_mc_dl2_data_file(
            str(file), "width>0", "software", "simple"
        )
        assert "pointing_alt" in data.colnames
        assert "theta" in data.colnames
        assert "true_source_fov_offset" in data.colnames
        assert data["true_energy"].unit == "TeV"
        assert point[0] >= 0
        assert point[0] <= 90


def test_load_mc_dl2_data_file_cut(p_dl2, gamma_dl2):
    """
    Check on quality cuts
    """
    dl2_mc = [p for p in gamma_dl2.glob("*")] + [p for p in p_dl2.glob("*")]
    for file in dl2_mc:
        data, _, _ = load_mc_dl2_data_file(
            str(file), "gammaness>0.1", "software", "simple"
        )
        assert np.all(data["gammaness"] > 0.1)


def test_load_mc_dl2_data_file_opt(p_dl2, gamma_dl2):
    """
    Check on event_type
    """
    dl2_mc = [p for p in gamma_dl2.glob("*")] + [p for p in p_dl2.glob("*")]
    for file in dl2_mc:
        data_s, _, _ = load_mc_dl2_data_file(
            str(file), "width>0", "software", "simple"
        )
        assert np.all(data_s["combo_type"] > 0)  # TODO: change in next PR


def test_load_mc_dl2_data_file_exc(p_dl2, gamma_dl2):
    """
    Check on event_type exceptions
    """
    dl2_mc = [p for p in gamma_dl2.glob("*")] + [p for p in p_dl2.glob("*")]
    for file in dl2_mc:
        event_type = "abc"
        with pytest.raises(
            ValueError,
            match=f"Unknown event type '{event_type}'.",
        ):
            _, _, _ = load_mc_dl2_data_file(str(file), "width>0", event_type, "simple")


def test_get_dl2_mean_mc(p_dl2, gamma_dl2):
    """
    Check on MC DL2
    """
    dl2_mc = [p for p in gamma_dl2.glob("*")] + [p for p in p_dl2.glob("*")]
    for file in dl2_mc:
        event_data = pd.read_hdf(str(file), key="events/parameters")
        event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
        event_data.sort_index(inplace=True)
        events = get_dl2_mean(event_data)
        assert "true_energy" in events.columns
        assert events["multiplicity"].dtype == int


def test_get_dl2_mean_avg(dl2_test):
    """
    Check on average evaluation
    """
    event_data = pd.read_hdf(str(dl2_test), key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)
    events = get_dl2_mean(event_data)
    assert np.allclose(np.array(events["gammaness"]), np.array([0.5, 0.6, 1]))


def test_get_dl2_mean_exc(p_dl2, gamma_dl2):
    """
    Check on exceptions (weight type)
    """
    dl2_mc = [p for p in gamma_dl2.glob("*")] + [p for p in p_dl2.glob("*")]
    for file in dl2_mc:
        weight = "abc"
        event_data = pd.read_hdf(str(file), key="events/parameters")
        event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
        event_data.sort_index(inplace=True)
        with pytest.raises(ValueError, match=f"Unknown weight type '{weight}'."):
            _ = get_dl2_mean(event_data, weight_type=weight)


def test_load_irf_files(IRF):
    """
    Check on IRF dictionaries
    """

    irf, header = load_irf_files(str(IRF))
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
                "file_names",
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


def test_load_irf_files_exc(temp_irf_exc):
    """
    Check on exception (FileNotFound)
    """
    with pytest.raises(
        FileNotFoundError,
        match="Could not find any IRF data files in the input directory.",
    ):
        _, _ = load_irf_files(str(temp_irf_exc))


def test_load_lst_dl1_data_file(dl1_lst):
    """
    Check on LST DL1
    """
    for file in dl1_lst:
        events, _ = load_lst_dl1_data_file(str(file))
        assert "event_type" in events.columns
        assert "slope" in events.columns
        assert "az_tel" not in events.columns
        events = events.reset_index()
        s = events.duplicated(subset=["obs_id_lst", "event_id_lst"])
        s1 = ~s
        assert s1.all()


def test_load_magic_dl1_data_files(merge_magic, config_gen):
    """
    Check on MAGIC DL1
    """

    events, _ = load_magic_dl1_data_files(str(merge_magic), config_gen)
    assert list(events.index.names) == ["obs_id_magic", "event_id_magic", "tel_id"]
    assert "event_id" not in events.columns
    events = events.reset_index()
    s = events.duplicated(subset=["obs_id_magic", "event_id_magic", "tel_id"])
    s1 = ~s
    assert s1.all()


def test_load_magic_dl1_data_files_exc(temp_DL1_M_exc, config_gen):
    """
    Check on MAGIC DL1: exceptions (no DL1 files)
    """
    with pytest.raises(
        FileNotFoundError,
        match="Could not find any DL1 data files in the input directory.",
    ):
        _, _ = load_magic_dl1_data_files(str(temp_DL1_M_exc), config_gen)


def test_get_stereo_events_data(coincidence_stereo, config_gen):
    """
    Check on stereo data reading
    """

    for file in coincidence_stereo.glob("*"):
        event_data = pd.read_hdf(str(file), key="events/parameters")
        event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
        event_data.sort_index(inplace=True)
        data = get_stereo_events(event_data, config_gen)
        assert np.all(data["multiplicity"] > 1)
        assert np.all(data["combo_type"] >= 0)


def test_get_stereo_events_data_cut(coincidence_stereo, config_gen):
    """
    Check on quality cuts
    """

    for file in coincidence_stereo.glob("*"):
        event_data = pd.read_hdf(str(file), key="events/parameters")
        event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
        event_data.sort_index(inplace=True)
        data = get_stereo_events(event_data, config_gen, "intensity>50")
        assert np.all(data["intensity"] > 50)


def test_load_dl2_data_file(real_dl2):
    """
    Checks on default loading
    """
    for file in real_dl2.glob("*"):
        data, on, dead = load_dl2_data_file(str(file), "width>0", "software", "simple")
        assert "pointing_alt" in data.colnames
        assert "timestamp" in data.colnames
        assert data["reco_energy"].unit == "TeV"
        assert on.unit == "s"
        assert on > 0
        assert dead > 0


def test_load_dl2_data_file_cut(real_dl2):
    """
    Check on quality cuts
    """
    for file in real_dl2.glob("*"):
        data, _, _ = load_dl2_data_file(
            str(file), "gammaness<0.9", "software", "simple"
        )
        assert np.all(data["gammaness"] < 0.9)


def test_load_dl2_data_file_opt(real_dl2):
    """
    Check on event_type
    """
    for file in real_dl2.glob("*"):
        data_s, _, _ = load_dl2_data_file(str(file), "width>0", "software", "simple")
        assert np.all(data_s["combo_type"] > 0)  # TODO: change in next PR


def test_load_dl2_data_file_exc(real_dl2):
    """
    Check on event_type exceptions
    """
    for file in real_dl2.glob("*"):
        event_type = "abc"
        with pytest.raises(
            ValueError,
            match=f"Unknown event type '{event_type}'.",
        ):
            _, _, _ = load_dl2_data_file(str(file), "width>0", event_type, "simple")


def test_get_dl2_mean_real(real_dl2):
    """
    Check on real data DL2
    """
    for file in real_dl2.glob("*"):
        event_data = pd.read_hdf(str(file), key="events/parameters")
        event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
        event_data.sort_index(inplace=True)
        events = get_dl2_mean(event_data)
        assert "timestamp" in events.columns


def test_index(real_index):
    """
    Check on DL3 creation (up to indexes)
    """
    print("Indexes created")
