import re
import sys
import xlrd
import datetime
import argparse
import numpy as np
from astropy import units as u


def get_obs_ids_from_column(col_obs_id):

    obs_ids_total = []

    for cell in col_obs_id:

        if type(cell) == np.str_:

            list_tmp = cell.split(',')

            for string in list_tmp:

                if '-' in string:
                    edges = string.split('-')
                    obs_id_list = np.arange(float(edges[0]), float(edges[1]) + 0.1, 1).tolist()
                    obs_ids_total += obs_id_list
                else:
                    obs_id = float(re.findall('(\d\d+)', string)[0])
                    obs_ids_total.append(obs_id)
        else:
            obs_id = float(cell)
            obs_ids_total.append(obs_id)

    obs_ids_total = np.array(obs_ids_total).astype(int)

    return obs_ids_total


def get_common_data_info(
    input_file, source_name, period_start=None, period_end=None, skip_obs_ids_lst=[], skip_obs_ids_magic=[]):

    # --- load the input excel file ---
    wb = xlrd.open_workbook(input_file)
    sheet = wb.sheets()[0]

    col_date_lst = np.array(sheet.col_values(0))[9:]
    col_source = np.array(sheet.col_values(1))[9:]
    col_obs_time = np.array(sheet.col_values(6))[9:]
    col_obs_time_wobble = np.array(sheet.col_values(7))[9:]
    col_obs_id_lst = np.array(sheet.col_values(5))[9:]
    col_obs_id_magic = np.array(sheet.col_values(10))[9:]

    # --- extract the information of the input source name ---
    print(f'\nThe input source name: {source_name}')

    names_list = np.unique(col_source)

    if source_name not in names_list:
        print(f'\nThe source name "{source_name}" does NOT exist in the input common observation data list. '
              f'Select one of the following name:\n{names_list}\n\nExiting.\n')
        sys.exit()

    condition = (col_source == source_name)

    col_date_lst = col_date_lst[condition]
    col_obs_time = col_obs_time[condition]
    col_obs_time_wobble = col_obs_time_wobble[condition]
    col_obs_id_lst = col_obs_id_lst[condition]
    col_obs_id_magic = col_obs_id_magic[condition]

    # --- extract the data taken within the input period ---
    if (period_start != None) or (period_end != None):

        print(f'\nTime period where the common observation data are extracted: {[period_start, period_end]}')

        condition = np.repeat(True, len(col_date_lst))

        if (period_start != None):

            times_unix = [datetime.datetime.strptime(date, '%Y_%m_%d').timestamp() for date in col_date_lst]
            start_time_unix = datetime.datetime.strptime(period_start, '%Y_%m_%d').timestamp()

            condition_start = np.array(times_unix) >= start_time_unix
            condition = (condition & condition_start)

        if (period_end != None):

            times_unix = [datetime.datetime.strptime(date, '%Y_%m_%d').timestamp() for date in col_date_lst]
            end_time_unix = datetime.datetime.strptime(period_end, '%Y_%m_%d').timestamp()

            condition_end = np.array(times_unix) <= end_time_unix
            condition = (condition & condition_end)

        col_date_lst = col_date_lst[condition]
        col_obs_time = col_obs_time[condition]
        col_obs_time_wobble = col_obs_time_wobble[condition]
        col_obs_id_lst = col_obs_id_lst[condition]
        col_obs_id_magic = col_obs_id_magic[condition]

    if len(col_date_lst) == 0:
        print('--> No observation dates are found within the input time period. Exiting.\n')
        sys.exit()

    print(f'--> Found the data in the following dates (LST convention):\n{col_date_lst}')

    # --- check the joint observation time ---
    total_obs_time = col_obs_time.astype(float).sum() * u.min
    total_obs_time_wobble = col_obs_time_wobble.astype(float).sum() * u.min

    print(f'\nTotal joint observation time = {total_obs_time.to(u.hour).value:.1f} [h]')
    print(f'(with same wobble = {total_obs_time_wobble.to(u.hour).value:.1f} [h])')

    # --- extract the observation IDs ---
    obs_ids_lst = get_obs_ids_from_column(col_obs_id_lst)
    obs_ids_magic = get_obs_ids_from_column(col_obs_id_magic)

    if skip_obs_ids_lst != []:

        skip_obs_ids_lst = np.unique(skip_obs_ids_lst)
        obs_ids_lst = np.sort(list(set(obs_ids_lst) ^ set(skip_obs_ids_lst)))

        print(f'\nLST skip IDs: {skip_obs_ids_lst}')

    if skip_obs_ids_magic != []:

        skip_obs_ids_magic = np.unique(skip_obs_ids_magic)
        obs_ids_magic = np.sort(list(set(obs_ids_magic) ^ set(skip_obs_ids_magic)))

        print(f'MAGIC skip IDs: {skip_obs_ids_magic}')

    print(f'\nNumber of LST observation IDs = {len(obs_ids_lst)}')
    print(f'Number of MAGIC observation IDs = {len(obs_ids_magic)}')

    print('\nLST observation IDs:')
    print(*obs_ids_lst)

    print('\nMAGIC observation IDs:')
    print(*obs_ids_magic)


def main():

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str,
        help='Path to an input excel file of MAGIC-LST common observation data list.'
    )

    arg_parser.add_argument(
        '--source-name', '-n', dest='source_name', type=str,
        help='Source name that the common observation information are extracted.'
    )

    arg_parser.add_argument(
        '--period-start', '-s', dest='period_start', type=str, default=None,
        help='Start date of the time period where the common observation information are extracted.'
    )

    arg_parser.add_argument(
        '--period-end', '-e', dest='period_end', type=str, default=None,
        help='End date of the time period where the common observation information are extracted.'
    )

    arg_parser.add_argument(
        '--skip-obs-ids-lst', '-l', dest='skip_obs_ids_lst', nargs='*', type=int, default=[],
        help='LST observation IDs that are skipped in the output information.'
    )

    arg_parser.add_argument(
        '--skip-obs-ids-magic', '-m', dest='skip_obs_ids_magic', nargs='*', type=int, default=[],
        help='MAGIC observation IDs that are skipped in the output information.'
    )

    args = arg_parser.parse_args()

    get_common_data_info(
        args.input_file, args.source_name,
        args.period_start, args.period_end, args.skip_obs_ids_lst, args.skip_obs_ids_magic
    )

    print('\nDone.\n')


if __name__ == "__main__":
    main()
