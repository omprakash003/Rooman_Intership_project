import pandas as pd


df = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=[ 
    'unit_number', 'time_in_cycles', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3',
    'sensor_measurement_1', 'sensor_measurement_2', 'sensor_measurement_3', 'sensor_measurement_4',
    'sensor_measurement_5', 'sensor_measurement_6', 'sensor_measurement_7', 'sensor_measurement_8',
    'sensor_measurement_9', 'sensor_measurement_10', 'sensor_measurement_11', 'sensor_measurement_12',
    'sensor_measurement_13', 'sensor_measurement_14', 'sensor_measurement_15', 'sensor_measurement_16',
    'sensor_measurement_17', 'sensor_measurement_18', 'sensor_measurement_19', 'sensor_measurement_20',
    'sensor_measurement_21'
])


print("\nMissing values:\n", df.isnull().sum())


print("\nDuplicate rows:", df.duplicated().sum())


max_cycles = df.groupby('unit_number')['time_in_cycles'].max()
print("\nMax cycle per unit (first 5):\n", max_cycles.head())


print("\nSensor Summary Stats:\n", df.describe().T.iloc[3:, :5])  
