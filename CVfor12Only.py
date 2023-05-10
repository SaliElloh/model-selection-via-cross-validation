from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import math

e = math.e


train_data = np.loadtxt("train.csv")
test_data = np.loadtxt("test.csv")

# separate the data input and output values. split the sorted data into input and output, the input are the
# calendar-year values, and the output are the indicator of the working-age population

x_train, y_train = train_data[:, 0], train_data[:, 1]
x_test, y_test = test_data[:, 0], test_data[:, 1]

# Start Cross-Validation:

k=6
kf = KFold(n_splits=k, shuffle=False)


# define the range of alpha values to search over
alphas = [0, e**(-25), e**(-20), e**(-14), e**(-7), e**(-3), 1, e**3, e**7]

test_rmse = []

for alpha in alphas:

    test_rmse_fold = []

    # Loop over the folds
    for train_index, test_index in kf.split(x_train):
        # Split the data into training and testing sets for the current fold
        x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        # Define a scaler function that scales the data for the current fold
        def scaled_x(X):
            x_scaled = (X - np.mean(x_train_fold)) / np.std(x_train_fold)
            return x_scaled

        def scaled_y(Y):
            y_scaled = (Y - np.mean(y_train_fold)) / np.std(y_train_fold)
            return y_scaled

        def descaled_y(y_scaled):
            y_descaled = (y_scaled * np.std(y_train_fold)) + np.mean(y_train_fold)
            return y_descaled

        # Scale the training and testing data for the current fold
        x_train_scaled = scaled_x(x_train_fold)
        y_train_scaled = scaled_y(y_train_fold)
        x_test_scaled = scaled_x(x_test_fold)
        y_test_scaled = scaled_y(y_test_fold)

        # Import the Ridge regression class on the training data for the current fold
        model = Ridge(alpha=alpha, solver='cholesky')
        poly = PolynomialFeatures(degree=12)
        x_train_poly = poly.fit_transform(x_train_scaled.reshape(-1, 1))
        x_test_poly = poly.transform(x_test_scaled.reshape(-1, 1))

        # Fit the model to the training data for the current fold
        model.fit(x_train_poly, y_train_scaled.reshape(-1, 1))

        # Compute the testing MSE for the current fold
        y_test_pred_scaled = model.predict(x_test_poly)
        test_rmse_fold.append(math.sqrt(mean_squared_error(descaled_y(y_test_scaled), descaled_y(y_test_pred_scaled))))

        coefficients = model.coef_


        # Take the average RMSE value for this degree:
    test_rmse.append(np.mean(test_rmse_fold))

    a_star = np.argmin(test_rmse)

# Plot the average of testing MSE for each degree
test_rmse_average = np.average(test_rmse)

print(f"Average test RMSE for each Alpha: {test_rmse}")
print(f"Optimal Alpha, or Regularizer term: {alphas[a_star]}")

# plot the alphas vs the average test RMSE obtained

plt.plot(alphas, test_rmse, label='Testing RMSE')
plt.title("RMSE vs. Alpha of Polynomial")
plt.xlabel("Alpha of Polynomial")
plt.ylabel("RMSE")
plt.legend()
plt.show()
