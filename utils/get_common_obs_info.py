import re
import sys
import xlrd
import argparse
import numpy as np
from astropy import units as u


def get_obs_ids(col_obs_id):

    obs_ids_total = []

    for cell in col_obs_id:

        if type(cell) == np.str_:

            list_tmp = cell.split(',')

            for string in list_tmp:

                if '-' in string:

                    edges = string.split('-')
                    obs_id_list = np.arange(float(edges[0]), float(edges[-1]) + 0.1, 1).tolist()
                    obs_ids_total += obs_id_list
                    
                else:

                    obs_id = float(re.findall('(\d\d+)', string)[0])
                    obs_ids_total.append(obs_id)

        else:

            obs_id = float(cell)
            obs_ids_total.append(obs_id)

    obs_ids_total = np.array(obs_ids_total).astype(int)

    return obs_ids_total


def get_common_data_info(input_file, source_name, skip_obs_ids_lst=[], skip_obs_ids_magic=[]):
    
    # --- load the input excel file ---
    wb = xlrd.open_workbook(input_file)
    sheet = wb.sheets()[0]

    col_date_lst = np.array(sheet.col_values(0))[9:]
    col_source = np.array(sheet.col_values(1))[9:]
    col_obs_time = np.array(sheet.col_values(7))[9:]
    col_obs_id_lst = np.array(sheet.col_values(5))[9:]
    col_obs_id_magic = np.array(sheet.col_values(10))[9:]
    
    # --- extract information of the input source ---
    print(f'\nChecking common observation data of {source_name}...')

    names_list = np.unique(col_source)

    if source_name not in names_list:
        print(f'\nSource name "{source_name}" does NOT exist in the input common observation list. '
              f'Select one of the following name:\n{names_list}\n\nExiting.\n')
        sys.exit()

    condition = (col_source == source_name)

    col_date_lst = col_date_lst[condition]
    col_obs_time = col_obs_time[condition]
    col_obs_id_lst = col_obs_id_lst[condition]
    col_obs_id_magic = col_obs_id_magic[condition]

    print(f'\n--> Found the data taken in the following dates (LST convention):\n{col_date_lst}')

    col_obs_time = col_obs_time.astype(float)
    total_obs_time = col_obs_time.sum() * u.min

    print(f'\nTotal joint observation time = {total_obs_time.to(u.hour).value:.1f} [hour]')
    
    # --- extract observation IDs ---
    skip_obs_ids_lst = np.unique(skip_obs_ids_lst)
    skip_obs_ids_magic = np.unique(skip_obs_ids_magic)

    print(f'\nLST skip IDs: {skip_obs_ids_lst}')
    print(f'MAGIC skip IDs: {skip_obs_ids_magic}')

    obs_ids_lst = get_obs_ids(col_obs_id_lst)
    obs_ids_magic = get_obs_ids(col_obs_id_magic)

    obs_ids_output_lst = np.sort(list(set(obs_ids_lst) ^ set(skip_obs_ids_lst)))
    obs_ids_output_magic = np.sort(list(set(obs_ids_magic) ^ set(skip_obs_ids_magic)))

    print(f'\nNumber of LST observation runs = {len(obs_ids_output_lst)}/{len(obs_ids_lst)}')
    print(f'Number of MAGIC observation runs = {len(obs_ids_output_magic)}/{len(obs_ids_magic)}')

    print(f'\nLST observation IDs:')
    print(*obs_ids_output_lst)

    print(f'\nMAGIC observation IDs:')
    print(*obs_ids_output_magic)

                                
def main():

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str, 
        help='Input excel file of MAGIC and LST common observation list'    
    )

    arg_parser.add_argument(
        '--source-name', '-n', dest='source_name', type=str, 
        help='Source name of which you want to extract the observation info'
    )

    arg_parser.add_argument(
        '--skip-obs-ids-lst', '-l', dest='skip_obs_ids_lst', nargs='*', type=int, default=[], 
        help='LST Observation IDs that are skipped in the output list'
    )

    arg_parser.add_argument(
        '--skip-obs-ids-magic', '-m', dest='skip_obs_ids_magic', nargs='*', type=int, default=[], 
        help='MAGIC Observation IDs that are skipped in the output list'
    )

    args = arg_parser.parse_args()

    get_common_data_info(
        args.input_file, args.source_name, args.skip_obs_ids_lst, args.skip_obs_ids_magic
    )

    print('\nDone.\n')


if __name__ == "__main__":
    main()

