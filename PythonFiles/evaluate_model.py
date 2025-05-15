import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import joblib

model = load_model('models/lstm_rul_model.h5')
X_test = np.load('data/X_test.npy')
y_test_scaled = np.load('data/y_test.npy').reshape(-1, 1)

scaler_y = joblib.load('models/rul_scaler.pkl')

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test = scaler_y.inverse_transform(y_test_scaled)

mse = mean_squared_error(y_test, y_pred)
print(f"Test Loss (MSE): {mse}")

np.save('models/y_pred.npy', y_pred.flatten())
print("Predictions (first 10):", y_pred.flatten()[:10])
print("Actual y_test (first 10):", y_test.flatten()[:10])
