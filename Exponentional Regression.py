import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Path to the CSV file
file_path = r'D:\COLLEGE\SMT 4\Metnum\Student_Performance.csv'

# Read the CSV file with delimiter ';'
data = pd.read_csv(file_path, delimiter=';')

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

def test_regression():
    assert isinstance(rms_error_exponential, float), "Invalid linear RMS error"
    assert len(y_pred_exponential) == len(NL), "Linear prediction does not match the number of data points"
    print("All tests passed.")

test_regression()

# Plotting the graph
plt.figure(figsize=(8, 6))
plt.scatter(NL, NT, label='Original Data')
plt.plot(NL, y_pred_exponential, color='green', label='Exponential Regression')
plt.xlabel('Number of Exercises (NL)', fontsize=12)
plt.ylabel('Test Scores (NT)', fontsize=12)
plt.legend()
plt.title('Relationship between Number of Exercises and Test Scores\n(Exponential Regression)\n', fontsize= 14)
plt.show()


