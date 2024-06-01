import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# Path to the CSV file
file_path = r'D:\COLLEGE\SMT 4\Metnum\Student_Performance.csv'

# Read the CSV file with delimiter ';'
data = pd.read_csv(file_path, delimiter=';')


# Assuming columns in CSV are 'NL' for Number of Exercises and 'NT' for Test Scores
NL = data['NL']
NT = data['NT']

# Exponential Regression Model
X = NL.values.reshape(-1, 1)
y = NT.values
log_y = np.log(y)
exponential_model = LinearRegression()
exponential_model.fit(X, log_y)
log_y_pred = exponential_model.predict(X)
y_pred_exponential = np.exp(log_y_pred)
rms_error_exponential = np.sqrt(np.mean((y - y_pred_exponential) ** 2))
print("RMS Error (Exponential Regression):", rms_error_exponential)

def test_regression_exponential():
    assert isinstance(rms_error_exponential, float), "Invalid linear RMS error"
    assert len(y_pred_exponential) == len(NL), "Linear prediction does not match the number of data points"
    print("All Exponential Regression tests passed.")

test_regression_exponential()

# Define the linear function
def linear_function(x, m, c):
    return m * x + c

popt, pcov = curve_fit(linear_function, NL, NT)

m, c = popt

y_pred_linear = linear_function(NL, m, c)
rms_error_linear = np.sqrt(np.mean((NT - y_pred_linear) ** 2))
print("RMS Error (Linear Regression):", rms_error_linear)

def test_regression_liner():
    assert isinstance(rms_error_linear, float), "Invalid linear RMS error"
    assert len(y_pred_linear) == len(NL), "Linear prediction does not match the number of data points"
    print("All Regerssion Linear tests passed.")

test_regression_liner()

# Plotting the graph
plt.scatter(NL, NT, label='Original Data')

# Exponential Regression Plot
plt.plot(NL, y_pred_exponential, color='green', label='Exponential Regression')

# Linear Regression Plot
plt.plot(NL, y_pred_linear, color='red', label='Linear Regression')

plt.xlabel('Number of Exercises (NL)')
plt.ylabel('Test Scores (NT)')
plt.legend()
plt.title('Relationship between Number of Exercises and Test Scores\n')
plt.show()

