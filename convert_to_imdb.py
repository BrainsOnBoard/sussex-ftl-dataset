#!/usr/bin/env python3
from glob import glob
import os
import pandas as pd

def convert_db(path):
    print('Converting ' + path + '...')
    df = pd.read_csv(os.path.join(path, 'training.csv'))

    # Strip whitespace from column headers
    df.rename(columns=lambda x: x.strip(), inplace=True)

    column_map = {'X': 'X [mm]', 'Y': 'Y [mm]', 'Z': 'Z [mm]',
                  'Rx': 'Heading [degrees]', 'Ry': 'Pitch [degrees]', 'Rz': 'Roll [degrees]',
                  'Time [s]': 'Timestamp [ms]'}
    df.rename(columns=column_map, inplace=True)

    # Convert timestamps from s to ms
    df['Timestamp [ms]'] *= 1000

    # Strip folder name from image file paths
    df['Filename'] = [os.path.basename(path) for path in df['Filename']]

    # These columns *must* be present for image databases, so if they aren't,
    # just write NaNs for them
    for col in ('X [mm]', 'Y [mm]', 'Z [mm]', 'Heading [degrees]'):
        if not col in df:
            # NB: Using an actual NaN value results in empty values being written
            df[col] = 'NaN'

    # Write out database_entries.csv
    df.to_csv(os.path.join(path, 'database_entries.csv'), index=False)

training_files = glob('**/training.csv', recursive=True)
for file in training_files:
    convert_db(os.path.dirname(file))
