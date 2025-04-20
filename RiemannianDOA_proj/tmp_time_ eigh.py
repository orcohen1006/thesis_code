import numpy as np
import matplotlib.pyplot as plt

# Parameters
k = 1 # Number of Taylor terms
x_vals = np.linspace(1e-3, 3, 1000)
true_log = np.log(x_vals)

# Taylor approximation of log(x) around x=1
approx_log = np.zeros_like(x_vals)
for n in range(1, k + 1):
    approx_log += ((-1)**(n + 1)) * ((x_vals - 1)**n) / n

# Error and bound
true_error = np.abs(true_log - approx_log)

xi = np.minimum(x_vals, 1)
error_bound = np.abs((x_vals - 1)**(k + 1)) / ((k + 1) * xi**(k + 1))
# error_bound = np.abs((x_vals - 1)**(k + 1) / (k + 1))

error_bound = (x_vals-1)**2 * (x_vals**2 + 1)/2
# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_vals, true_log, 'k-', label='log(x)', linewidth=2)
plt.plot(x_vals, approx_log, 'b--', label=f'g(x; {k})', linewidth=2)
plt.plot(x_vals, true_error, 'r-', label='True Error', linewidth=1.5)
plt.plot(x_vals, error_bound, 'g--', label='Error Bound', linewidth=1.5)
plt.xlabel('x')
plt.ylabel('Value')
plt.title(f'Taylor Approximation of log(x) with {k} Terms around x=1')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
