import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Path to the CSV file
file_path = r'D:\COLLEGE\SMT 4\Metnum\Student_Performance.csv'

# Read the CSV file with delimiter ';'
data = pd.read_csv(file_path, delimiter=';')

# Assuming columns in CSV are 'NL' for Number of Exercises and 'NT' for Test Scores
NL = data['NL']
NT = data['NT']

# Define the linear function
def linear_function(x, m, c):
    return m * x + c

popt, pcov = curve_fit(linear_function, NL, NT)

m, c = popt

y_pred_linear = linear_function(NL, m, c)
rms_error_linear = np.sqrt(np.mean((NT - y_pred_linear) ** 2))
print("RMS Error (Linear Regression):", rms_error_linear)

# Testing the code
def test_regression():
    assert isinstance(rms_error_linear, float), "Invalid linear RMS error"
    assert len(y_pred_linear) == len(NL), "Linear prediction does not match the number of data points"
    print("All tests passed.")

test_regression()

# Plotting the graph
plt.figure(figsize=(8, 6))
plt.scatter(NL, NT, label='Original Data')
plt.plot(NL, y_pred_linear, color='red', label='Linear Regression')
plt.xlabel('Number of Exercises (NL)')
plt.ylabel('Test Scores (NT)')
plt.legend()
plt.title('Relationship between Number of Exercises and Test Scores\n(Linear Regression)\n', fontsize=14)
plt.show()

