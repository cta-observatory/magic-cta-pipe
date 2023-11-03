import pytest
from ctapipe.calib import CameraCalibrator
from ctapipe.io import EventSource
from traitlets.config import Config

from magicctapipe.image import MAGICClean
from magicctapipe.image.calib import calibrate


@pytest.fixture(scope="session")
def tel_id_LST():
    return 1


@pytest.fixture(scope="session")
def tel_id_MAGIC():
    return 2


def test_calibrate_LST(dl0_gamma, config_calib, tel_id_LST):
    assigned_tel_ids = [1, 2, 3]
    for input_file in dl0_gamma:
        event_source = EventSource(
            input_file, allowed_tels=assigned_tel_ids, focal_length_choice="effective"
        )

        obs_id = event_source.obs_ids[0]

        subarray = event_source.subarray

        tel_descriptions = subarray.tel
        camera_geoms = {}

        for tel_id, telescope in tel_descriptions.items():
            camera_geoms[tel_id] = telescope.camera.geometry

        config_lst = config_calib["LST"]

        extractor_type_lst = config_lst["image_extractor"].pop("type")
        config_extractor_lst = {extractor_type_lst: config_lst["image_extractor"]}

        calibrator_lst = CameraCalibrator(
            image_extractor_type=extractor_type_lst,
            config=Config(config_extractor_lst),
            subarray=subarray,
        )

        for event in event_source:
            if (event.count < 200) and (tel_id_LST in event.trigger.tels_with_trigger):
                signal_pixels, image, peak_time = calibrate(
                    event=event,
                    tel_id=tel_id_LST,
                    obs_id=obs_id,
                    config=config_lst,
                    camera_geoms=camera_geoms,
                    calibrator=calibrator_lst,
                    is_lst=True,
                )

                assert len(signal_pixels) == 1855
                assert signal_pixels.dtype == bool
                assert len(image) == 1855
                assert len(peak_time) == 1855

        config_lst["image_extractor"]["type"] = extractor_type_lst


def test_calibrate_MAGIC(dl0_gamma, config_calib, tel_id_MAGIC):
    assigned_tel_ids = [1, 2, 3]
    for input_file in dl0_gamma:
        event_source = EventSource(
            input_file, allowed_tels=assigned_tel_ids, focal_length_choice="effective"
        )

        subarray = event_source.subarray

        tel_descriptions = subarray.tel
        camera_geoms = {}

        for tel_id, telescope in tel_descriptions.items():
            camera_geoms[tel_id] = telescope.camera.geometry

        config_magic = config_calib["MAGIC"]
        config_magic["magic_clean"].update({"find_hotpixels": False})

        extractor_type_magic = config_magic["image_extractor"].pop("type")
        config_extractor_magic = {extractor_type_magic: config_magic["image_extractor"]}
        magic_clean = {}
        for k in [1, 2]:
            magic_clean[k] = MAGICClean(camera_geoms[k], config_magic["magic_clean"])
        calibrator_magic = CameraCalibrator(
            image_extractor_type=extractor_type_magic,
            config=Config(config_extractor_magic),
            subarray=subarray,
        )

        for event in event_source:
            if (event.count < 200) and (
                tel_id_MAGIC in event.trigger.tels_with_trigger
            ):
                signal_pixels, image, peak_time = calibrate(
                    event=event,
                    tel_id=tel_id_MAGIC,
                    config=config_magic,
                    magic_clean=magic_clean,
                    calibrator=calibrator_magic,
                    is_lst=False,
                )

                assert len(signal_pixels) == 1039
                assert signal_pixels.dtype == bool
                assert len(image) == 1039
                assert len(peak_time) == 1039

        config_magic["image_extractor"]["type"] = extractor_type_magic


def test_calibrate_exc_1(dl0_gamma, config_calib, tel_id_MAGIC):
    assigned_tel_ids = [1, 2, 3]
    for input_file in dl0_gamma:
        event_source = EventSource(
            input_file, allowed_tels=assigned_tel_ids, focal_length_choice="effective"
        )
        subarray = event_source.subarray
        config_magic = config_calib["MAGIC"]
        config_magic["magic_clean"].update({"find_hotpixels": False})
        extractor_type_magic = config_magic["image_extractor"].pop("type")
        config_extractor_magic = {extractor_type_magic: config_magic["image_extractor"]}
        calibrator_magic = CameraCalibrator(
            image_extractor_type=extractor_type_magic,
            config=Config(config_extractor_magic),
            subarray=subarray,
        )

        for event in event_source:
            if (event.count < 200) and (
                tel_id_MAGIC in event.trigger.tels_with_trigger
            ):
                with pytest.raises(
                    ValueError,
                    match="Check the provided parameters and the telescope type; MAGIC calibration not possible if magic_clean not provided",
                ):
                    _, _, _ = calibrate(
                        event=event,
                        tel_id=tel_id_MAGIC,
                        config=config_magic,
                        calibrator=calibrator_magic,
                        is_lst=False,
                    )
        config_magic["image_extractor"]["type"] = extractor_type_magic


def test_calibrate_exc_2(dl0_gamma, config_calib, tel_id_LST):
    assigned_tel_ids = [1, 2, 3]
    for input_file in dl0_gamma:
        event_source = EventSource(
            input_file, allowed_tels=assigned_tel_ids, focal_length_choice="effective"
        )

        subarray = event_source.subarray

        tel_descriptions = subarray.tel
        camera_geoms = {}

        for tel_id, telescope in tel_descriptions.items():
            camera_geoms[tel_id] = telescope.camera.geometry

        config_lst = config_calib["LST"]

        extractor_type_lst = config_lst["image_extractor"].pop("type")
        config_extractor_lst = {extractor_type_lst: config_lst["image_extractor"]}

        calibrator_lst = CameraCalibrator(
            image_extractor_type=extractor_type_lst,
            config=Config(config_extractor_lst),
            subarray=subarray,
        )

        for event in event_source:
            if (event.count < 200) and (tel_id_LST in event.trigger.tels_with_trigger):
                with pytest.raises(
                    ValueError,
                    match="Check the provided parameters and the telescope type; LST calibration not possible if obs_id not provided",
                ):
                    _, _, _ = calibrate(
                        event=event,
                        tel_id=tel_id_LST,
                        config=config_lst,
                        camera_geoms=camera_geoms,
                        calibrator=calibrator_lst,
                        is_lst=True,
                    )
        config_lst["image_extractor"]["type"] = extractor_type_lst


def test_calibrate_exc_3(dl0_gamma, config_calib, tel_id_LST):
    assigned_tel_ids = [1, 2, 3]
    for input_file in dl0_gamma:
        event_source = EventSource(
            input_file, allowed_tels=assigned_tel_ids, focal_length_choice="effective"
        )

        obs_id = event_source.obs_ids[0]

        subarray = event_source.subarray

        config_lst = config_calib["LST"]

        extractor_type_lst = config_lst["image_extractor"].pop("type")
        config_extractor_lst = {extractor_type_lst: config_lst["image_extractor"]}

        calibrator_lst = CameraCalibrator(
            image_extractor_type=extractor_type_lst,
            config=Config(config_extractor_lst),
            subarray=subarray,
        )

        for event in event_source:
            if (event.count < 200) and (tel_id_LST in event.trigger.tels_with_trigger):
                with pytest.raises(
                    ValueError,
                    match="Check the provided parameters and the telescope type; LST calibration not possible if gamera_geoms not provided",
                ):
                    signal_pixels, image, peak_time = calibrate(
                        event=event,
                        tel_id=tel_id_LST,
                        obs_id=obs_id,
                        config=config_lst,
                        calibrator=calibrator_lst,
                        is_lst=True,
                    )
        config_lst["image_extractor"]["type"] = extractor_type_lst


def test_calibrate_exc_4(dl0_gamma, config_calib, tel_id_MAGIC):
    assigned_tel_ids = [1, 2, 3]
    for input_file in dl0_gamma:
        event_source = EventSource(
            input_file, allowed_tels=assigned_tel_ids, focal_length_choice="effective"
        )
        subarray = event_source.subarray
        tel_descriptions = subarray.tel
        magic_clean = {}

        for tel_id in range(len(tel_descriptions.items())):
            magic_clean[tel_id] = f"camera {tel_id}"
        config_magic = config_calib["MAGIC"]
        config_magic["magic_clean"].update({"find_hotpixels": False})
        extractor_type_magic = config_magic["image_extractor"].pop("type")
        config_extractor_magic = {extractor_type_magic: config_magic["image_extractor"]}
        calibrator_magic = CameraCalibrator(
            image_extractor_type=extractor_type_magic,
            config=Config(config_extractor_magic),
            subarray=subarray,
        )

        for event in event_source:
            if (event.count < 200) and (
                tel_id_MAGIC in event.trigger.tels_with_trigger
            ):
                with pytest.raises(
                    ValueError,
                    match="Check the provided magic_clean parameter; MAGIC calibration not possible if magic_clean not a dictionary of MAGICClean objects",
                ):
                    _, _, _ = calibrate(
                        event=event,
                        tel_id=tel_id_MAGIC,
                        config=config_magic,
                        calibrator=calibrator_magic,
                        magic_clean=magic_clean,
                        is_lst=False,
                    )
        config_magic["image_extractor"]["type"] = extractor_type_magic


def test_calibrate_exc_5(dl0_gamma, config_calib, tel_id_LST):
    assigned_tel_ids = [1, 2, 3]
    for input_file in dl0_gamma:
        event_source = EventSource(
            input_file, allowed_tels=assigned_tel_ids, focal_length_choice="effective"
        )

        obs_id = event_source.obs_ids[0]

        subarray = event_source.subarray

        tel_descriptions = subarray.tel
        camera_geoms = {}

        for tel_id in range(len(tel_descriptions.items())):
            camera_geoms[tel_id] = f"camera {tel_id}"

        config_lst = config_calib["LST"]

        extractor_type_lst = config_lst["image_extractor"].pop("type")
        config_extractor_lst = {extractor_type_lst: config_lst["image_extractor"]}

        calibrator_lst = CameraCalibrator(
            image_extractor_type=extractor_type_lst,
            config=Config(config_extractor_lst),
            subarray=subarray,
        )

        for event in event_source:
            if (event.count < 200) and (tel_id_LST in event.trigger.tels_with_trigger):
                with pytest.raises(
                    ValueError,
                    match="Check the provided camera_geoms parameter; LST calibration not possible if camera_geoms not a dictionary of CameraGeometry objects",
                ):
                    _, _, _ = calibrate(
                        event=event,
                        tel_id=tel_id_LST,
                        obs_id=obs_id,
                        config=config_lst,
                        camera_geoms=camera_geoms,
                        calibrator=calibrator_lst,
                        is_lst=True,
                    )
        config_lst["image_extractor"]["type"] = extractor_type_lst
