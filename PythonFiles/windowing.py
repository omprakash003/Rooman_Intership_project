import numpy as np
import pandas as pd


train_df = pd.read_csv('data/train_normalized.csv')
test_df = pd.read_csv('data/test_normalized.csv')


window_size = 50

def create_windows(df, window_size):
    sequences = []
    labels = []


    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit].reset_index(drop=True)

        if len(unit_data) >= window_size + 1:  # Ensure enough data points
            for start in range(len(unit_data) - window_size):
                end = start + window_size
                sequence = unit_data.iloc[start:end].drop(['unit_number', 'time_in_cycles', 'RUL'], axis=1).values
                label = unit_data.iloc[end]['RUL']  # RUL after the window
                sequences.append(sequence)
                labels.append(label)

 
    X = np.array(sequences)
    y = np.array(labels)
    return X, y

X_train, y_train = create_windows(train_df, window_size)
X_test, y_test = create_windows(test_df, window_size)


np.save('data/X_train.npy', X_train)
np.save('data/y_train.npy', y_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_test.npy', y_test)


print("Windowing complete.")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
