import glob

import numpy as np
import pandas as pd
import pytest

from magicctapipe.io.io import (
    get_dl2_mean,
    get_stereo_events,
    load_dl2_data_file,
    load_irf_files,
    load_magic_dl1_data_files,
    load_mc_dl2_data_file,
    load_train_data_files_tel,
)


@pytest.mark.dependency()
def test_exist_dl1_mc(gamma_l1_monly, p_l1_monly):
    """
    Check if DL1 MC produced
    """

    assert len(glob.glob(f"{gamma_l1_monly}/*")) == 4
    assert len(glob.glob(f"{p_l1_monly}/*")) == 2


@pytest.mark.dependency(depends=["test_exist_dl1_mc"])
def test_exist_dl1_stereo_mc(gamma_stereo_monly, p_stereo_monly):
    """
    Check if DL1 stereo MC produced
    """

    assert len(glob.glob(f"{gamma_stereo_monly[0]}/*")) == 2
    assert len(glob.glob(f"{gamma_stereo_monly[1]}/*")) == 2
    assert len(glob.glob(f"{p_stereo_monly[0]}/*")) == 1
    assert len(glob.glob(f"{p_stereo_monly[1]}/*")) == 1


@pytest.mark.dependency(depends=["test_exist_dl1_stereo_mc"])
class TestStereoMC:
    def test_get_stereo_events_mc(self, gamma_stereo_monly, p_stereo_monly, config_gen):
        """
        Check on stereo data reading
        """

        stereo_mc = (
            [p for p in gamma_stereo_monly[0].glob("*")]
            + [p for p in gamma_stereo_monly[1].glob("*")]
            + [p for p in p_stereo_monly[0].glob("*")]
            + [p for p in p_stereo_monly[1].glob("*")]
        )

        for file in stereo_mc:
            event_data = pd.read_hdf(str(file), key="events/parameters")
            event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
            event_data.sort_index(inplace=True)
            data = get_stereo_events(event_data, config_gen)
            assert np.all(data["multiplicity"] == 2)
            assert np.all(data["combo_type"] == 3)

    def test_get_stereo_events_mc_cut(
        self, gamma_stereo_monly, p_stereo_monly, config_gen
    ):
        """
        Check on quality cuts
        """
        stereo_mc = (
            [p for p in gamma_stereo_monly[0].glob("*")]
            + [p for p in gamma_stereo_monly[1].glob("*")]
            + [p for p in p_stereo_monly[0].glob("*")]
            + [p for p in p_stereo_monly[1].glob("*")]
        )
        for file in stereo_mc:
            event_data = pd.read_hdf(str(file), key="events/parameters")
            event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
            event_data.sort_index(inplace=True)
            data = get_stereo_events(event_data, config_gen, "intensity>50")
            assert np.all(data["intensity"] > 50)
            assert len(data) > 0

    def test_load_train_data_files_tel(
        self, p_stereo_monly, gamma_stereo_monly, config_gen
    ):
        """
        Check dictionary
        """

        for stereo in [p_stereo_monly, gamma_stereo_monly]:
            events = load_train_data_files_tel(str(stereo[0]), config_gen)
            assert list(events.keys()) == [2, 3]
            data = events[2]
            assert "off_axis" in data.columns
            assert "true_event_class" not in data.columns

    def test_load_train_data_files_tel_off(self, gamma_stereo_monly, config_gen):
        """
        Check off-axis cut
        """
        events = load_train_data_files_tel(
            str(gamma_stereo_monly[0]),
            config=config_gen,
            offaxis_min="0.2 deg",
            offaxis_max="0.5 deg",
        )
        data = events[2]
        assert np.all(data["off_axis"] >= 0.2)
        assert np.all(data["off_axis"] <= 0.5)
        assert len(data) > 0


@pytest.mark.dependency(depends=["test_exist_dl1_stereo_mc"])
def test_exist_rf(RF_monly):
    """
    Check if RFs produced
    """

    assert len(glob.glob(f"{RF_monly}/*")) == 6


@pytest.mark.dependency(depends=["test_exist_rf"])
def test_exist_dl2_mc(p_dl2_monly, gamma_dl2_monly):
    """
    Check if DL2 MC produced
    """

    assert len(glob.glob(f"{p_dl2_monly}/*")) == 1
    assert len(glob.glob(f"{gamma_dl2_monly}/*")) == 1


@pytest.mark.dependency(depends=["test_exist_dl2_mc"])
class TestDL2MC:
    def test_load_mc_dl2_data_file(self, config_gen, p_dl2_monly, gamma_dl2_monly):
        """
        Checks on default loading
        """
        dl2_mc = [p for p in gamma_dl2_monly.glob("*")] + [
            p for p in p_dl2_monly.glob("*")
        ]
        for file in dl2_mc:
            data, point, _ = load_mc_dl2_data_file(
                config_gen, str(file), "width>0", "magic_only", "simple"
            )
            assert "pointing_alt" in data.colnames
            assert "theta" in data.colnames
            assert "true_source_fov_offset" in data.colnames
            assert data["true_energy"].unit == "TeV"
            assert point[0] >= 0
            assert point[0] <= 90

    def test_load_mc_dl2_data_file_cut(self, config_gen, p_dl2_monly, gamma_dl2_monly):
        """
        Check on quality cuts
        """
        dl2_mc = [p for p in gamma_dl2_monly.glob("*")] + [
            p for p in p_dl2_monly.glob("*")
        ]
        for file in dl2_mc:
            data, _, _ = load_mc_dl2_data_file(
                config_gen, str(file), "gammaness>0.1", "magic_only", "simple"
            )
            assert np.all(data["gammaness"] > 0.1)
            assert len(data) > 0

    def test_load_mc_dl2_data_file_opt(self, config_gen, p_dl2_monly, gamma_dl2_monly):
        """
        Check on event_type
        """
        dl2_mc = [p for p in gamma_dl2_monly.glob("*")] + [
            p for p in p_dl2_monly.glob("*")
        ]
        for file in dl2_mc:

            data_m, _, _ = load_mc_dl2_data_file(
                config_gen, str(file), "width>0", "magic_only", "simple"
            )

            assert np.all(data_m["combo_type"] == 3)
            assert len(data_m) > 0

    def test_load_mc_dl2_data_file_exc(self, config_gen, p_dl2_monly, gamma_dl2_monly):
        """
        Check on event_type exceptions
        """
        dl2_mc = [p for p in gamma_dl2_monly.glob("*")] + [
            p for p in p_dl2_monly.glob("*")
        ]
        for file in dl2_mc:
            event_type = "abc"
            with pytest.raises(
                ValueError,
                match=f"Unknown event type '{event_type}'.",
            ):
                _, _, _ = load_mc_dl2_data_file(
                    config_gen, str(file), "width>0", event_type, "simple"
                )

    def test_get_dl2_mean_mc(self, p_dl2_monly, gamma_dl2_monly):
        """
        Check on MC DL2
        """
        dl2_mc = [p for p in gamma_dl2_monly.glob("*")] + [
            p for p in p_dl2_monly.glob("*")
        ]
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

    def test_get_dl2_mean_exc(self, p_dl2_monly, gamma_dl2_monly):
        """
        Check on exceptions (weight type)
        """
        dl2_mc = [p for p in gamma_dl2_monly.glob("*")] + [
            p for p in p_dl2_monly.glob("*")
        ]
        for file in dl2_mc:
            weight = "abc"
            event_data = pd.read_hdf(str(file), key="events/parameters")
            event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
            event_data.sort_index(inplace=True)
            with pytest.raises(ValueError, match=f"Unknown weight type '{weight}'."):
                _ = get_dl2_mean(event_data, weight_type=weight)


@pytest.mark.dependency(depends=["test_exist_dl2_mc"])
def test_exist_irf(IRF_monly):
    """
    Check if IRFs produced
    """

    assert len(glob.glob(f"{IRF_monly}/*")) == 1


@pytest.mark.dependency(depends=["test_exist_irf"])
class TestIRF:
    def test_load_irf_files(self, IRF_monly):
        """
        Check on IRF dictionaries
        """

        irf, header = load_irf_files(str(IRF_monly))
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
        assert header["EVT_TYPE"] == "magic_only"


@pytest.mark.dependency()
def test_exist_dl1_magic(M2_l1_monly, M1_l1_monly):
    """
    Check if DL1 created
    """

    assert len(glob.glob(f"{M1_l1_monly}/*")) == 2
    assert len(glob.glob(f"{M2_l1_monly}/*")) == 2


@pytest.mark.dependency(depends=["test_exist_dl1_magic"])
def test_exist_merged_magic(merge_magic_monly):
    """
    Check if MAGIC merged
    """

    assert len(glob.glob(f"{merge_magic_monly}/*")) == 1


@pytest.mark.dependency(depends=["test_exist_merged_magic"])
class TestDL1MAGIC:
    def test_load_magic_dl1_data_files(self, merge_magic_monly, config_gen):
        """
        Check on MAGIC DL1
        """

        events, _ = load_magic_dl1_data_files(str(merge_magic_monly), config_gen)
        assert list(events.index.names) == ["obs_id_magic", "event_id_magic", "tel_id"]
        assert "event_id" not in events.columns
        events = events.reset_index()
        s = events.duplicated(subset=["obs_id_magic", "event_id_magic", "tel_id"])
        s1 = ~s
        assert s1.all()


@pytest.mark.dependency(depends=["test_exist_merged_magic"])
def test_exist_stereo(stereo_monly):
    """
    Check if MAGIC stereo exist
    """

    assert len(glob.glob(f"{stereo_monly}/*")) == 1


@pytest.mark.dependency(depends=["test_exist_stereo"])
class TestStereoData:
    def test_get_stereo_events_data(self, stereo_monly, config_gen):
        """
        Check on stereo data reading
        """

        for file in stereo_monly.glob("*"):
            event_data = pd.read_hdf(str(file), key="events/parameters")
            event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
            event_data.sort_index(inplace=True)
            data = get_stereo_events(event_data, config_gen)
            assert np.all(data["multiplicity"] == 2)
            assert np.all(data["combo_type"] == 3)

    def test_get_stereo_events_data_cut(self, stereo_monly, config_gen):
        """
        Check on quality cuts
        """

        for file in stereo_monly.glob("*"):
            event_data = pd.read_hdf(str(file), key="events/parameters")
            event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
            event_data.sort_index(inplace=True)
            data = get_stereo_events(event_data, config_gen, "intensity>50")
            assert np.all(data["intensity"] > 50)
            assert len(data) > 0


@pytest.mark.dependency(depends=["test_exist_stereo", "test_exist_rf"])
def test_exist_dl2(real_dl2_monly):
    """
    Check if DL2 exist
    """

    assert len(glob.glob(f"{real_dl2_monly}/*")) == 1


@pytest.mark.dependency(depends=["test_exist_dl2"])
class TestDL2Data:
    def test_load_dl2_data_file(self, config_gen, real_dl2_monly):
        """
        Checks on default loading
        """
        for file in real_dl2_monly.glob("*"):
            data, on, dead = load_dl2_data_file(
                config_gen, str(file), "width>0", "magic_only", "simple"
            )
            assert "pointing_alt" in data.colnames
            assert "timestamp" in data.colnames
            assert data["reco_energy"].unit == "TeV"
            assert on.unit == "s"
            assert on > 0
            assert dead > 0

    def test_load_dl2_data_file_cut(self, config_gen, real_dl2_monly):
        """
        Check on quality cuts
        """
        for file in real_dl2_monly.glob("*"):
            data, _, _ = load_dl2_data_file(
                config_gen, str(file), "gammaness<0.9", "magic_only", "simple"
            )
            assert np.all(data["gammaness"] < 0.9)
            assert len(data) > 0

    def test_load_dl2_data_file_opt(self, config_gen, real_dl2_monly):
        """
        Check on event_type
        """
        for file in real_dl2_monly.glob("*"):

            data_m, _, _ = load_dl2_data_file(
                config_gen, str(file), "width>0", "magic_only", "simple"
            )

            assert np.all(data_m["combo_type"] == 3)
            assert len(data_m) > 0

    def test_load_dl2_data_file_exc(self, config_gen, real_dl2_monly):
        """
        Check on event_type exceptions
        """
        for file in real_dl2_monly.glob("*"):
            event_type = "abc"
            with pytest.raises(
                ValueError,
                match=f"Unknown event type '{event_type}'.",
            ):
                _, _, _ = load_dl2_data_file(
                    config_gen, str(file), "width>0", event_type, "simple"
                )

    def test_get_dl2_mean_real(self, real_dl2_monly):
        """
        Check on real data DL2
        """
        for file in real_dl2_monly.glob("*"):
            event_data = pd.read_hdf(str(file), key="events/parameters")
            event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
            event_data.sort_index(inplace=True)
            events = get_dl2_mean(event_data)
            assert "timestamp" in events.columns


@pytest.mark.dependency(depends=["test_exist_dl2", "test_exist_irf"])
def test_exist_dl3(real_dl3_monly):
    """
    Check if DL3 exist
    """

    assert len(glob.glob(f"{real_dl3_monly}/dl3*")) == 1


@pytest.mark.dependency(depends=["test_exist_dl3"])
def test_exist_index(real_index_monly):
    """
    Check if indexes created
    """

    assert len(glob.glob(f"{real_index_monly}/*index*")) == 2
