import glob

import astropy.table
import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import yaml
from ctapipe_io_lst import REFERENCE_LOCATION

from magicctapipe.io.io import (
    check_input_list,
    format_object,
    get_custom_cuts,
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


class TestGeneral:
    def test_check_input_list(self):
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

    def test_get_custom_cuts(self, config):
        with open(config, "rb") as f:
            config = yaml.safe_load(f)
        # Create energy bins
        config_eng_bins = config["create_irf"]["energy_bins"]
        eng_bins_start = u.Quantity(config_eng_bins["start"])
        eng_bins_stop = u.Quantity(config_eng_bins["stop"])

        energy_bins = u.TeV * np.geomspace(
            start=eng_bins_start.to_value("TeV").round(3),
            stop=eng_bins_stop.to_value("TeV").round(3),
            num=config_eng_bins["n_edges"],
        )

        # Load the cuts configs. 'cut_type' will be ignored
        config_gh_irf = config["create_irf"]["gammaness"]
        config_theta_irf = config["create_irf"]["theta"]

        # Get the custom cuts without interpolating on energy_bins
        cut_table_gh = get_custom_cuts(config_gh_irf, energy_bins)
        cut_table_theta = get_custom_cuts(config_theta_irf, energy_bins)
        for cut_table in (cut_table_gh, cut_table_theta):
            assert isinstance(cut_table, astropy.table.QTable)
            assert cut_table["low"].unit == "TeV"
            assert cut_table["high"].unit == "TeV"
            assert cut_table["center"].unit == "TeV"
        assert cut_table_gh["cut"].unit == u.dimensionless_unscaled
        assert cut_table_gh["cut"][15] == 0.5
        assert cut_table_theta["cut"][15] == 0.3 * u.deg

        # Get the custom cuts with linear interpolating on energy_bins
        config_gh_irf["interpolate_kind"] = "linear"
        cut_table_gh = get_custom_cuts(config_gh_irf, energy_bins)
        assert np.isclose(cut_table_gh["cut"][15].value, 0.5074, atol=0.0001)

    def test_telescope_combinations(self, config_gen, config_gen_4lst):
        """
        Simple check on telescope combinations
        """
        M_LST, M_LST_comb = telescope_combinations(config_gen)
        LSTs, LSTs_comb = telescope_combinations(config_gen_4lst)
        assert M_LST == {1: "LST-1", 2: "MAGIC-I", 3: "MAGIC-II"}
        assert M_LST_comb == {
            "LST-1_MAGIC-I": [1, 2],
            "LST-1_MAGIC-I_MAGIC-II": [1, 2, 3],
            "LST-1_MAGIC-II": [1, 3],
            "MAGIC-I_MAGIC-II": [2, 3],
        }
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

    def test_format_object(self):
        """
        Simple check on a string
        """
        str_a = "a{b[[xz,cde}"
        str_b = format_object(str_a)
        assert str_b == "a b  xzcde "

    def test_save_pandas_data_in_table(self, temp_pandas, pd_test):
        """
        Check on pandas dataframe (before = after saving it)
        """
        out = temp_pandas / "pandas.h5"
        save_pandas_data_in_table(pd_test, str(out), "abc", "event")
        df1 = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=["a", "b"])
        df = pd.read_hdf(str(out), key="event")
        assert df.equals(df1)


@pytest.mark.dependency()
def test_exist_dl1_mc(gamma_l1, p_l1):
    """
    Check if DL1 MC produced
    """

    assert len(glob.glob(f"{gamma_l1}/*")) == 4
    assert len(glob.glob(f"{p_l1}/*")) == 2


@pytest.mark.dependency(depends=["test_exist_dl1_mc"])
def test_conc_mc(gamma_l1):
    """
    Check if DL1 MC files have computed concentration
    """
    for file in glob.glob(f"{gamma_l1}/*"):
        d = pd.read_hdf(file, "/events/parameters")
        # out of 3 concentration parameter the only sanity check we can do if pixel one is > 0
        # while it is unlikely to have value > 1 it is also possible,
        # and the other two concentration parameters in particular can be <=0 or > 1
        assert np.all(d.groupby("tel_id").min()[["concentration_pixel"]] > 0)


@pytest.mark.dependency(depends=["test_exist_dl1_mc"])
def test_exist_dl1_stereo_mc(gamma_stereo, p_stereo):
    """
    Check if DL1 stereo MC produced
    """

    assert len(glob.glob(f"{gamma_stereo[0]}/*")) == 2
    assert len(glob.glob(f"{gamma_stereo[1]}/*")) == 2
    assert len(glob.glob(f"{p_stereo[0]}/*")) == 1
    assert len(glob.glob(f"{p_stereo[1]}/*")) == 1


@pytest.mark.dependency(depends=["test_exist_dl1_stereo_mc"])
class TestStereoMC:
    def test_get_stereo_events_mc(self, gamma_stereo, p_stereo, config_gen):
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

    def test_get_stereo_events_mc_cut(self, gamma_stereo, p_stereo, config_gen):
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
            assert len(data) > 0

    def test_load_train_data_files(self, p_stereo, gamma_stereo):
        """
        Check dictionary of the combo types
        """

        for stereo in [p_stereo, gamma_stereo]:
            events = load_train_data_files(str(stereo[0]))
            assert list(events.keys()) == ["LST1_M1", "LST1_M1_M2", "LST1_M2", "M1_M2"]
            data = events["LST1_M1"]
            assert np.all(data["combo_type"] == 0)
            assert "off_axis" in data.columns
            assert "true_event_class" not in data.columns

    def test_load_train_data_files_off(self, gamma_stereo):
        """
        Check off-axis cut
        """
        events = load_train_data_files(
            str(gamma_stereo[0]), offaxis_min="0.2 deg", offaxis_max="0.5 deg"
        )
        data = events["LST1_M1"]
        assert np.all(data["off_axis"] >= 0.2)
        assert np.all(data["off_axis"] <= 0.5)
        assert len(data) > 0

    def test_load_train_data_files_exc(self, temp_train_exc):
        """
        Check on exceptions
        """

        with pytest.raises(
            FileNotFoundError,
            match="Could not find any DL1-stereo data files in the input directory.",
        ):
            _ = load_train_data_files(str(temp_train_exc))

    def test_load_train_data_files_tel(self, p_stereo, gamma_stereo, config_gen):
        """
        Check dictionary
        """

        for stereo in [p_stereo, gamma_stereo]:
            events = load_train_data_files_tel(str(stereo[0]), config_gen)
            assert list(events.keys()) == [1, 2, 3]
            data = events[2]
            assert "off_axis" in data.columns
            assert "true_event_class" not in data.columns

    def test_load_train_data_files_tel_off(self, gamma_stereo, config_gen):
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
        assert len(data) > 0

    def test_load_train_data_files_tel_exc(self, temp_train_exc, config_gen):
        """
        Check on exceptions
        """
        with pytest.raises(
            FileNotFoundError,
            match="Could not find any DL1-stereo data files in the input directory.",
        ):
            _ = load_train_data_files(str(temp_train_exc), config_gen)


@pytest.mark.dependency(depends=["test_exist_dl1_stereo_mc"])
def test_exist_rf(RF):
    """
    Check if RFs produced
    """

    assert len(glob.glob(f"{RF}/*")) == 12


@pytest.mark.dependency(depends=["test_exist_rf"])
def test_exist_dl2_mc(p_dl2, gamma_dl2):
    """
    Check if DL2 MC produced
    """

    assert len(glob.glob(f"{p_dl2}/*")) == 1
    assert len(glob.glob(f"{gamma_dl2}/*")) == 1


@pytest.mark.dependency(depends=["test_exist_dl2_mc"])
class TestDL2MC:
    def test_load_mc_dl2_data_file(self, p_dl2, gamma_dl2):
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

    def test_load_mc_dl2_data_file_cut(self, p_dl2, gamma_dl2):
        """
        Check on quality cuts
        """
        dl2_mc = [p for p in gamma_dl2.glob("*")] + [p for p in p_dl2.glob("*")]
        for file in dl2_mc:
            data, _, _ = load_mc_dl2_data_file(
                str(file), "gammaness>0.1", "software", "simple"
            )
            assert np.all(data["gammaness"] > 0.1)
            assert len(data) > 0

    def test_load_mc_dl2_data_file_opt(self, p_dl2, gamma_dl2):
        """
        Check on event_type
        """
        dl2_mc = [p for p in gamma_dl2.glob("*")] + [p for p in p_dl2.glob("*")]
        for file in dl2_mc:
            data_s, _, _ = load_mc_dl2_data_file(
                str(file), "width>0", "software", "simple"
            )
            assert np.all(data_s["combo_type"] < 3)
            assert len(data_s) > 0

    def test_load_mc_dl2_data_file_exc(self, p_dl2, gamma_dl2):
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
                _, _, _ = load_mc_dl2_data_file(
                    str(file), "width>0", event_type, "simple"
                )

    def test_get_dl2_mean_mc(self, p_dl2, gamma_dl2):
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

    def test_get_dl2_mean_avg(self, dl2_test):
        """
        Check on average evaluation
        """
        event_data = pd.read_hdf(str(dl2_test), key="events/parameters")
        event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
        event_data.sort_index(inplace=True)
        events = get_dl2_mean(event_data)
        assert np.allclose(np.array(events["gammaness"]), np.array([0.5, 0.6, 1]))

    def test_get_dl2_mean_exc(self, p_dl2, gamma_dl2):
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


@pytest.mark.dependency(depends=["test_exist_dl2_mc"])
def test_exist_irf(IRF):
    """
    Check if IRFs produced
    """

    assert len(glob.glob(f"{IRF}/*")) == 1


@pytest.mark.dependency(depends=["test_exist_irf"])
class TestIRF:
    def test_load_irf_files(self, IRF):
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

    def test_load_irf_files_exc(self, temp_irf_exc):
        """
        Check on exception (FileNotFound)
        """
        with pytest.raises(
            FileNotFoundError,
            match="Could not find any IRF data files in the input directory.",
        ):
            _, _ = load_irf_files(str(temp_irf_exc))


class TestDL1LST:
    def test_load_lst_dl1_data_file_old(self, dl1_lst_old):
        """
        Check on LST DL1
        """
        events, subarray = load_lst_dl1_data_file(str(dl1_lst_old))
        assert "event_type" in events.columns
        assert "slope" in events.columns
        assert "az_tel" not in events.columns
        events = events.reset_index()
        s = events.duplicated(subset=["obs_id_lst", "event_id_lst"])
        assert np.all(s == False)
        assert subarray.name == "LST-1 subarray"
        assert subarray.reference_location == REFERENCE_LOCATION

    @pytest.mark.dependency()
    def test_load_lst_dl1_data_file(self, dl1_lst):
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
            assert np.all(s == False)


@pytest.mark.dependency()
def test_exist_dl1_magic(M2_l1, M1_l1):
    """
    Check if DL1 created
    """

    assert len(glob.glob(f"{M1_l1}/*")) == 2
    assert len(glob.glob(f"{M2_l1}/*")) == 2


@pytest.mark.dependency(depends=["test_exist_dl1_magic"])
def test_exist_merged_magic(merge_magic):
    """
    Check if MAGIC merged
    """

    assert len(glob.glob(f"{merge_magic}/*")) == 1


@pytest.mark.dependency(depends=["test_exist_merged_magic"])
def test_conc_data(merge_magic):
    """
    Check if DL1 data files have computed concentration
    """
    for file in glob.glob(f"{str(merge_magic)}/*"):
        d = pd.read_hdf(file, "/events/parameters")
        assert np.all(d.groupby("tel_id").min()[["concentration_pixel"]] > 0)


@pytest.mark.dependency(depends=["test_exist_merged_magic"])
class TestDL1MAGIC:
    def test_load_magic_dl1_data_files(self, merge_magic, config_gen):
        """
        Check on MAGIC DL1
        """

        events, _ = load_magic_dl1_data_files(str(merge_magic), config_gen)
        assert list(events.index.names) == ["obs_id_magic", "event_id_magic", "tel_id"]
        assert "event_id" not in events.columns
        events = events.reset_index()
        s = events.duplicated(subset=["obs_id_magic", "event_id_magic", "tel_id"])
        assert np.all(s == False)

    def test_load_magic_dl1_data_files_exc(self, temp_DL1_M_exc, config_gen):
        """
        Check on MAGIC DL1: exceptions (no DL1 files)
        """
        with pytest.raises(
            FileNotFoundError,
            match="Could not find any DL1 data files in the input directory.",
        ):
            _, _ = load_magic_dl1_data_files(str(temp_DL1_M_exc), config_gen)


@pytest.mark.dependency(
    depends=["test_exist_merged_magic", "TestDL1LST::test_load_lst_dl1_data_file"]
)
def test_exist_coincidence(coincidence):
    """
    Check if coincidence created
    """

    assert len(glob.glob(f"{coincidence}/*")) == 1


@pytest.mark.dependency(depends=["test_exist_coincidence"])
def test_exist_coincidence_stereo(coincidence_stereo):
    """
    Check if coincidence stereo created
    """

    assert len(glob.glob(f"{coincidence_stereo}/*")) == 1


def test_get_stereo_events_multimatch(config_gen):
    """
    Check if multiple matched events are removed
    """
    d = {
        "obs_id": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "tel_id": [1, 2, 1, 1, 3, 1, 1, 2, 3],
        "event_id": [1, 1, 2, 2, 2, 3, 3, 3, 3],
    }
    event_data = pd.DataFrame(data=d)
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    data = get_stereo_events(event_data, config_gen)
    assert np.all(data["multiplicity"] == [2, 2, 2, 2])


@pytest.mark.dependency(depends=["test_exist_coincidence_stereo"])
class TestStereoData:
    def test_get_stereo_events_data(self, coincidence_stereo, config_gen):
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

    def test_get_stereo_events_data_cut(self, coincidence_stereo, config_gen):
        """
        Check on quality cuts
        """

        for file in coincidence_stereo.glob("*"):
            event_data = pd.read_hdf(str(file), key="events/parameters")
            event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
            event_data.sort_index(inplace=True)
            data = get_stereo_events(event_data, config_gen, "intensity>50")
            assert np.all(data["intensity"] > 50)
            assert len(data) > 0


@pytest.mark.dependency(depends=["test_exist_coincidence_stereo", "test_exist_rf"])
def test_exist_dl2(real_dl2):
    """
    Check if DL2 exist
    """

    assert len(glob.glob(f"{real_dl2}/*")) == 1


@pytest.mark.dependency(depends=["test_exist_dl2"])
class TestDL2Data:
    def test_load_dl2_data_file(self, real_dl2):
        """
        Checks on default loading
        """
        for file in real_dl2.glob("*"):
            data, on, dead = load_dl2_data_file(
                str(file), "width>0", "software", "simple"
            )
            assert "pointing_alt" in data.colnames
            assert "timestamp" in data.colnames
            assert data["reco_energy"].unit == "TeV"
            assert on.unit == "s"
            assert on > 0
            assert dead > 0

    def test_load_dl2_data_file_cut(self, real_dl2):
        """
        Check on quality cuts
        """
        for file in real_dl2.glob("*"):
            data, _, _ = load_dl2_data_file(
                str(file), "gammaness<0.9", "software", "simple"
            )
            assert np.all(data["gammaness"] < 0.9)
            assert len(data) > 0

    def test_load_dl2_data_file_opt(self, real_dl2):
        """
        Check on event_type
        """
        for file in real_dl2.glob("*"):
            data_s, _, _ = load_dl2_data_file(
                str(file), "width>0", "software", "simple"
            )
            assert np.all(data_s["combo_type"] < 3)
            assert len(data_s) > 0

    def test_load_dl2_data_file_exc(self, real_dl2):
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

    def test_get_dl2_mean_real(self, real_dl2):
        """
        Check on real data DL2
        """
        for file in real_dl2.glob("*"):
            event_data = pd.read_hdf(str(file), key="events/parameters")
            event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
            event_data.sort_index(inplace=True)
            events = get_dl2_mean(event_data)
            assert "timestamp" in events.columns


@pytest.mark.dependency(depends=["test_exist_dl2", "test_exist_irf"])
def test_exist_dl3(real_dl3):
    """
    Check if DL3 exist
    """

    assert len(glob.glob(f"{real_dl3}/dl3*")) == 1


@pytest.mark.dependency(depends=["test_exist_dl3"])
def test_exist_index(real_index):
    """
    Check if indexes created
    """

    assert len(glob.glob(f"{real_index}/*index*")) == 2
