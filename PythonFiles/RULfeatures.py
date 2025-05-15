import pandas as pd


columns = [
    'unit_number', 'time_in_cycles', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3',
    'sensor_measurement_1', 'sensor_measurement_2', 'sensor_measurement_3', 'sensor_measurement_4',
    'sensor_measurement_5', 'sensor_measurement_6', 'sensor_measurement_7', 'sensor_measurement_8',
    'sensor_measurement_9', 'sensor_measurement_10', 'sensor_measurement_11', 'sensor_measurement_12',
    'sensor_measurement_13', 'sensor_measurement_14', 'sensor_measurement_15', 'sensor_measurement_16',
    'sensor_measurement_17', 'sensor_measurement_18', 'sensor_measurement_19', 'sensor_measurement_20',
    'sensor_measurement_21'
]


df = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=columns)


rul_df = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
rul_df.columns = ['unit_number', 'max_cycle']
df = df.merge(rul_df, on='unit_number', how='left')
df['RUL'] = df['max_cycle'] - df['time_in_cycles']
df.drop('max_cycle', axis=1, inplace=True)


drop_columns = [
    'operational_setting_2', 'operational_setting_3',
    'sensor_measurement_1', 'sensor_measurement_5',
    'sensor_measurement_6', 'sensor_measurement_10',
    'sensor_measurement_16', 'sensor_measurement_18',
    'sensor_measurement_19'
]
df.drop(columns=drop_columns, inplace=True)


df.to_csv('data/labeled_cleaned.csv', index=False)


print("Cleaned dataset saved successfully.")
print("Shape after cleaning:", df.shape)
