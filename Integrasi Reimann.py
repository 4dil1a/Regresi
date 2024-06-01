import numpy as np
import time

def riemann_integral(f, a, b, N):
    dx = (b - a) / N
    x = np.linspace(a + dx/2, b - dx/2, N)  # Titik tengah untuk metode Riemann
    integral = np.sum(f(x)) * dx
    return integral

def f(x):
    return 4 / (1 + x**2)

def rms_error(estimated_pi, true_pi):
    return np.sqrt(np.mean((estimated_pi - true_pi)**2))

# Nilai referensi pi
true_pi = 3.14159265358979323846

# Variasi nilai N
N_values = [10, 100, 1000, 10000]

for N in N_values:
    start_time = time.time()
    estimated_pi = riemann_integral(f, 0, 1, N)
    end_time = time.time()
    error = rms_error(estimated_pi, true_pi)
    execution_time = end_time - start_time
    
    print(f"N = {N}")
    print(f"Estimated Pi: {estimated_pi}")
    print(f"RMS Error: {error}")
    print(f"Execution Time: {execution_time} seconds")
    print("-" * 30)
