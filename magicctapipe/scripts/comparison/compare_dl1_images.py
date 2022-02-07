import argparse
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
    clean_mask_1 = []

    source_1 = EventSource(input_url=input_file_1)

    for event in source_1:
        event_ids_1.append(event.index.event_id)
        images_1.append(event.dl1.tel[1].image)
        clean_mask_1.append(event.dl1.tel[1].image_mask)

    event_ids_2 = []
    images_2 = []
    clean_mask_2 = []

    source_2 = EventSource(input_url=input_file_2)

    for event in source_2:
        event_ids_2.append(event.index.event_id)
        images_2.append(event.dl1.tel[1].image)
        clean_mask_2.append(event.dl1.tel[1].image_mask)

    if event_ids_1 != event_ids_2:
        print("The two files have different event ids.")
