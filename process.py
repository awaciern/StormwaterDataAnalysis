# Importing libraries
import pandas as pd
from os import listdir
from os.path import isfile, join

# Get all the sensor dataset files
files = [f for f in listdir('../Data/Source') if isfile(join('../Data/Source', f))]

# Allocate initial big dataset to store all data in
date_range = pd.date_range(start='4/1/2018', end='12/1/2018 23:45', freq='15min')
data_big = pd.DataFrame(index=date_range,
                        columns=[f.strip('.csv') for f in files])
print(data_big)

# Loop through all the source datasets
for f in files:
    # Read in the data from the sensor
    data_sensor = pd.read_csv('../Data/Source/{0}'.format(f), index_col='Reading')
    data_sensor.index = pd.to_datetime(data_sensor.index)
    print(data_sensor)

    # Copy the sensor data to the appropriate field in the big dataset
    prev_date = pd.to_datetime('4/1/2018 0:00')
    for date in data_big.index:
        if date in data_sensor.index:
            data_big[f.strip('.csv')][date] = data_sensor['Value'][date]
        else:
             data_big[f.strip('.csv')][date] = data_big[f.strip('.csv')][prev_date]
        prev_date = date
    print(data_big[f.strip('.csv')])

# Save the initial big dataset
print(data_big)
data_big.to_csv('../Data/Big/Initial.csv')

# Allocate another big dataset that now has fields for flood events
data_big_flood = pd.DataFrame(data_big,
                              columns=[col for col in data_big]
                                      + ['A_Flood', 'B_Flood', 'C_Flood',
                                         'D_Flood', 'E_Flood',
                                         'X_Flood', 'Y_Flood', 'Z_Flood'])
print(data_big_flood)

# Function to mark a flood event on a site given a threshold
def mark_flood(site, threshold):
    stage = '{0}_Stage'.format(site)
    flood = '{0}_Flood'.format(site)
    for date in data_big_flood.index:
        if data_big[stage][date] > threshold:
            data_big_flood[flood][date] = 1
        else:
            data_big_flood[flood][date] = 0
    print(data_big_flood[[stage, flood]])

# Make function calls to mark flood events based on site thresholds
mark_flood('A', 16)
mark_flood('B', 16)
mark_flood('C', 15)
mark_flood('D', 11)
mark_flood('E', 16)
mark_flood('X', 7)
mark_flood('Y', 4)
mark_flood('Z', 6)

# Save the new big dataset with flood data
print(data_big_flood)
data_big_flood.to_csv('../Data/Big/Flood.csv')
