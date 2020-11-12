from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import operator

# Univariate polynomial regression
''' When we have 1 x parameter (based on which we should predict results)
Linear Regression = x_0 * theta_0 + x_1*theta_1 '''


# Path of data file
path = "B29.csv"
# Reading data from file
data = pd.read_csv(path, nrows=10000)

# Choose your x parameter (in this case we have one parameter)
x_1 = data["Depth"]


'''If we choose more than one independent var (multiple x) then
concatenate multiple columns to one x_1 var
x_1 = pd.DataFrame(np.c_[df['col1'], df['col2']..])'''

# Choose your y (value to predict)
y = data["Temperature"]


# This is another test data  to check how our model works
# boston_data = load_boston()
# data = pd.DataFrame(data=boston_data["data"], columns=boston_data["feature_names"])
# y = pd.DataFrame(boston_data["target"], columns=["Price in $1000s"])
# x_1 = data["LSTAT"]


# Make x_1 2D array
x_1 = np.array(x_1)[:, np.newaxis]


# Initialize array with mean squared error to measure difference between actual and predicted value
mse_array = []


# Function to fit polynomial regression


def linearRegression(degree):
    polynomial_features = PolynomialFeatures(degree=degree)
    # Create polynomial combinations of x_1 (if degree = 3: x_1^0,x_1^1...x_1^3)
    x_poly = polynomial_features.fit_transform(x_1)
    # Choose model
    model = LinearRegression()
    # Fit polynomial combinations of x_1 and values of y to model
    model.fit(x_poly, y)
    # Make predictions
    y_poly_pred = model.predict(x_poly)
    # Calculate mse
    mse = (mean_squared_error(y, y_poly_pred))
    # Append mse for each degree
    mse_array.append(mse)
    return y_poly_pred, polynomial_features, model


# Function to plot polynomial regression and predict values based on test data


def plot_predict_linearRegression(test_array):
    degree = 0
    while True:
        # Start with degree = 0
        linearRegression(degree)
        # Each time check if the next mean squared error is greater than previous,
        # if error increases then stop while loop
        if mse_array[len(mse_array) - 1] > mse_array[len(mse_array) - 2]:
            break
        else:
            # if error decreases continue increasing degree
            degree += 1
    # do not include the last degree (degree with greater mse than previous)
    degree -= 1
    print("The degree that fits the model with decreased error is ", degree)
    # Plot our data (original x and y values)
    plt.scatter(x_1, y, s=10)
    # sort the values of x and y before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x_1, linearRegression(degree)[0]), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    # Plot the regression (with predicted y values)
    plt.plot(x, y_poly_pred, color='m')
    plt.show()
    # Test your data and see predicted values
    test = linearRegression(degree)[2].predict(linearRegression(degree)[1].fit_transform(test_array))
    print("The predicated value(s) is (are): ",test)
    # The closer r2 to 1 the better
    print("R2 score is: ", r2_score(y, linearRegression(degree)[0]))
    # The lower mse the better
    print("MSE is: ", (mean_squared_error(y, linearRegression(degree)[0])))
    return degree


# Call function and enter test data
array = []
test_array = [float(s) for s in input('Enter numbers (separated by space) to test your data : ').split()]
for test in test_array:
    array.append([test])

plot_predict_linearRegression(array)




