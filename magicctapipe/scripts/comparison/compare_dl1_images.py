import sys
import argparse
import numpy as np

from ctapipe.io import EventSource


def parse_args(args):
    """
    Parse command line options and arguments.
    """

    parser = argparse.ArgumentParser(description="", prefix_chars='-')
    parser.add_argument("--input1", nargs='?', help="Path to first Dl1 file.")
    parser.add_argument("--input2", nargs='?', help="Path to second Dl1 file.")

    return parser.parse_args(args)


def compare_dl1_images(input_file_1, input_file_2):

    event_ids_1 = []
    images_1 = []
    clean_masks_1 = []

    source_1 = EventSource(input_url=input_file_1)

    for event in source_1:
        event_ids_1.append(event.index.event_id)
        images_1.append(event.dl1.tel[1].image)
        clean_masks_1.append(event.dl1.tel[1].image_mask)

    event_ids_2 = []
    images_2 = []
    clean_masks_2 = []

    source_2 = EventSource(input_url=input_file_2)

    for event in source_2:
        event_ids_2.append(event.index.event_id)
        images_2.append(event.dl1.tel[1].image)
        clean_masks_2.append(event.dl1.tel[1].image_mask)

    if event_ids_1 != event_ids_2:
        print("The two files have different event ids.")

    for event_i in range(len(event_ids_1)):
        assert event_ids_1[event_i] == event_ids_2[event_i]
        image_1 = np.array(images_1[event_i])
        image_2 = np.array(images_2[event_i])
        image_diff = np.absolute(image_1 - image_2)
        diff_gtr_0 = np.where(image_diff > 0.01)[0]
        if len(diff_gtr_0) > 0:
            print(f"Event ID {event_ids_1[event_i]} has different images.")
            print(image_diff[diff_gtr_0])

        clean_mask_1 = clean_masks_1[event_i]
        clean_mask_2 = clean_masks_2[event_i]
        if not np.array_equal(clean_mask_1, clean_mask_2):
            print(f"Event ID {event_ids_1[event_i]} has different cleaning masks.")
            clean_mask_diff = clean_mask_1 == clean_mask_2
            print(np.where(clean_mask_diff is False)[0])


def main(*args):
    flags = parse_args(args)

    input_1 = flags.input1
    input_2 = flags.input2

    compare_dl1_images(
        input_file_1=input_1,
        input_file_2=input_2
    )


if __name__ == '__main__':
    main(*sys.argv[1:])
