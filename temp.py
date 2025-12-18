# ===============================
# Linear Regression in Spyder
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = pd.read_csv('25RP19523.csv')

# 2. Define features and target
X = data[['Volume']]      # Independent variable
y = data['CO2']           # Dependent variable (1D is preferred)

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 6. Model evaluation
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print('Training MSE:', mse_train)
print('Testing MSE :', mse_test)
print('Training R2 :', r2_train)
print('Testing R2  :', r2_test)

# 7. Save the trained model
filename = '25RP19523_MODEL.sav'
pickle.dump(model, open(filename, 'wb'))
print('Model saved as', filename)

# 8. Visualization (Training data)
plt.figure()
plt.scatter(X_train, y_train, label='Training Data')
plt.plot(X_train, y_train_pred, label='Regression Line')
plt.xlabel('Volume')
plt.ylabel('CO2')
plt.title('Volume vs CO2 (Training Set)')
plt.legend()
plt.show()

# ===============================
# Load model and make prediction
# ===============================

loaded_model = pickle.load(open('25RP19523_MODEL.sav', 'rb'))

# Example prediction
new_volume = np.array([[25]])
prediction = loaded_model.predict(new_volume)

print(f"Predicted CO2 value for Volume = 25: {prediction[0]}")
