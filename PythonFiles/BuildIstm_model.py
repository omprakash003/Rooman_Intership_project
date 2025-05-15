import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy').reshape(-1, 1)
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy').reshape(-1, 1)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

input_shape = X_train.shape[1:]
batch_size = 64
epochs = 50

model = Sequential()
model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = model.fit(X_train, y_train_scaled, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test, y_test_scaled), callbacks=[early_stop])

model.save('models/lstm_rul_model.h5')

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

mse = np.mean((y_pred - y_test)**2)
print(f"Test Loss (MSE): {mse}")

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

print("Predictions (first 10):", y_pred[:10].flatten())
print("Actual y_test (first 10):", y_test[:10].flatten())
