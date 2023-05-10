from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import math


train_data = np.loadtxt("train.csv")
test_data = np.loadtxt("test.csv")

# separate the data input and output values. split the sorted data into input and output, the input are the
# calendar-year values, and the output are the indicator of the working-age population

x_train, y_train = train_data[:, 0], train_data[:, 1]
x_test, y_test = test_data[:, 0], test_data[:, 1]


# Start Cross-Validation:

k=6
kf = KFold(n_splits=k, shuffle=False)

# double check if these values are correct

#scale both the training data and the testing data based on the mean and standard deviation values calculated earlier


degrees = range(0, 12)

# Initialize the training and testing MSE for the current degree

test_rmse = []  # Rename the variable to avoid confusion (


for degree in degrees:

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
        model = Ridge(alpha=0, solver='cholesky')
        poly = PolynomialFeatures(degree)
        x_train_poly = poly.fit_transform(x_train_scaled.reshape(-1, 1))
        x_test_poly = poly.transform(x_test_scaled.reshape(-1, 1))

        # Fit the model to the training data for the current fold
        model.fit(x_train_poly, y_train_scaled.reshape(-1, 1))

        # Compute the testing RMSE for the current fold

        y_test_pred_scaled = model.predict(x_test_poly)
        test_rmse_fold.append(math.sqrt(mean_squared_error(descaled_y(y_test_scaled), descaled_y(y_test_pred_scaled))))

        # Take the average RMSE value for this degree:
    test_rmse.append(np.mean(test_rmse_fold))

    print(test_rmse)

    d_star = np.argmin(test_rmse)


# print the optimal degree found, d_star:
print(f"Optimal degree: {d_star}")

# print the testing RMSE for all degrees:
print(f"Average Testing RMSE for each degree:'\n'{test_rmse}")

plt.plot(degrees, test_rmse, label='Testing RMSE')
plt.title("RMSE vs. Degree of Polynomial")
plt.xlabel("Degree of Polynomial")
plt.ylabel("RMSE")
plt.legend()
plt.show()

