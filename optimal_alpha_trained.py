from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import math

e = math.e

# load training and test data
train_data = np.loadtxt("train.csv")
test_data = np.loadtxt("test.csv")

# separate input and output variables
x_train, y_train = train_data[:, :-1], train_data[:, -1]
x_test, y_test = test_data[:, :-1], test_data[:, -1]

# apply standardization to input variables

def scaled_x(X):
    x_scaled = (X - np.mean(x_train)) / np.std(x_train)
    return x_scaled


def scaled_y(Y):
    y_scaled = (Y - np.mean(y_train)) / np.std(y_train)
    return y_scaled


def descaled_y(y_scaled):
    y_descaled = (y_scaled * np.std(y_train)) + np.mean(y_train)
    return y_descaled


# Scale the training and testing data for the current fold
x_train_scaled = scaled_x(x_train)
y_train_scaled = scaled_y(y_train)
x_test_scaled = scaled_x(x_test)
y_test_scaled = scaled_y(y_test)


# Import the Ridge regression class on the training data
model = Ridge(alpha=(e**-3), solver='cholesky')
poly = PolynomialFeatures(degree=12)
x_train_poly = poly.fit_transform(x_train_scaled.reshape(-1, 1))
x_test_poly = poly.transform(x_test_scaled.reshape(-1, 1))

# fit the model to the training data
model.fit(x_train_poly, y_train_scaled.reshape(-1, 1))

# make predictions on the test data
y_test_pred = model.predict(x_test_poly)

# make predictions on the test data
y_train_pred = model.predict(x_train_poly)

# descale y_test_pred and y_test_scaled
descale_y_test_pred = descaled_y(y_test_pred)
descale_y_test= descaled_y(y_test_scaled)

# descale y_train_pred and y_train_scaled
descale_y_train_pred = descaled_y(y_train_pred)
descale_y_train= descaled_y(y_train_scaled)

# compute the weights of the optimal alpha:
weights = model.coef_

# compute the test RMSE:
test_rmse = math.sqrt(mean_squared_error(descale_y_test, descale_y_test_pred))

# compute the train RMSE:
train_rmse = math.sqrt(mean_squared_error(descale_y_train, descale_y_train_pred))

# find the coefficients of the optimal lamda/apha
weights = model.coef_

# descale the weights:
weights_descaled = descaled_y(weights.reshape(-1, 1))

#print results:
print(f"the Test RMSE for the optimal alpha (e^-3) on all data is: {test_rmse:.5f}")
print(f"the Train RMSE for the optimal alpha (e^-3) on all data is: {train_rmse:.5f}")
print(f"the coefficients for the optimal alpha (e^-3) on all data is \n {weights}")


# Plot the training data
plt.scatter(x_train, y_train, color='blue', label='Training data')


# Plot the testing data
plt.scatter(x_test, y_test, color='red', label='Testing data')

# Plot the predicted test values:
plt.scatter(x_test, descale_y_test_pred, color='green', label='Ridge regression model for Optimal Degree')

# Define a sequence of values for the x-axis (years 1968-2023)
x_seq = np.linspace(1968, 2023, num=100).reshape(-1, 1)

# Scale the x-axis using the same scaling function used for the training and test data
x_seq_scaled = scaled_x(x_seq)

# Apply the polynomial features transformation to the scaled x-axis using the same degree of polynomial as d∗
x_seq_poly = poly.transform(x_seq_scaled)

# Use the trained Ridge regression model with the optimal lambda value λ∗ to predict the y values for the transformed x-axis
y_seq_pred_scaled = model.predict(x_seq_poly)

# Descale the predicted y values to get the actual y values
y_seq_pred = descaled_y(y_seq_pred_scaled)

# Plot the resulting polynomial curve along with all the training data
plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.plot(x_seq, y_seq_pred, color='green', label=f'Polynomial curve for d*={poly.degree} and lambda*={model.alpha}')
plt.xlabel('Year')
plt.ylabel('Y')
plt.legend()
plt.show()

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()




