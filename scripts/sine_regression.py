import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %%
# Example data (replace with your data)
x = np.linspace(0, 10, 100)
y = (
    3 * np.sin(2 * x + 1) + 0.5 + np.random.normal(0, 0.2, size=len(x))
)  # Noisy sine wave


# Define sine function to fit
def sine_wave(x, A, B, C, D):
    return A * np.sin(B * x + C) + D


# Initial parameter guesses (important for convergence)
initial_guess = [3, 2, 1, 0.5]

# Curve fitting
params, _ = curve_fit(sine_wave, x, y, p0=initial_guess)

# Extract fitted parameters
A_fit, B_fit, C_fit, D_fit = params

# Generate fitted curve
y_fit = sine_wave(x, A_fit, B_fit, C_fit, D_fit)

# Plot results
plt.scatter(x, y, label="Data", color="gray", alpha=0.6)
plt.plot(
    x,
    y_fit,
    label=f"Fitted: A={A_fit:.2f}, B={B_fit:.2f}, C={C_fit:.2f}, D={D_fit:.2f}",
    color="red",
)
plt.legend()
plt.show()
