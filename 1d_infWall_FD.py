import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Define the analytical solution
def analytical_solution(x, ke, mu, c, alpha):
    I_0 = (1/ke) * np.exp(-c**2 / alpha**2)  # Boundary condition at x = 0
    
    term1 = I_0 * np.exp(-ke * x / mu)
    term2 = (alpha * np.sqrt(np.pi) / (2 * mu)) * np.exp(-ke / mu * (x - (alpha**2 * ke / (4 * mu) + c)))
    
    erf1 = erf(-ke * alpha / (2 * mu) + (c - x) / alpha)
    erf2 = erf(-ke * alpha / (2 * mu) + c / alpha)
    
    return term1 - term2 * (erf1 - erf2)

# Define parameters
Nx = 4000  # Number of grid points
x = np.linspace(0, 1, Nx)  # Spatial grid
dx = x[1] - x[0]  # Grid spacing

ke = 1.0  # Absorption coefficient
c = 0.5  # Source center
alpha = 0.02  # Source width
mu = 0.5  # Direction cosine (change sign to test positive and negative cases)

# Define source function
S = np.exp(-((x - c) ** 2) / alpha**2)

# Initialize intensity array
I = np.zeros(Nx)

# Apply boundary conditions
if mu > 0:
    I[0] = (1 / ke) * np.exp(-c**2 / alpha**2)
    # Upwind scheme for positive mu
    for i in range(1, Nx):
        I[i] = (I[i-1] + (dx / mu) * S[i-1]) / (1 + (dx / mu) * ke)
else:
    I[-1] = (1 / ke) * np.exp(-((1 - c) ** 2) / alpha**2)
    # Upwind scheme for negative mu
    for i in range(Nx - 2, -1, -1):
        I[i] = (I[i+1] - (dx / mu) * S[i+1]) / (1 - (dx / mu) * ke)

# Save solution to CSV file with mu and ke values in filename
filename = f'radiation_solution_mu_{mu:.2f}_ke_{ke:.2f}.csv'
solution_data = np.column_stack((x, I))
np.savetxt(filename, solution_data, delimiter=',', header='x,intensity', comments='')

# Compute the analytical solution
I_analytical = analytical_solution(x, ke, mu, c, alpha)

# Plot results
plt.plot(x, I, label=f'Intensity for mu = {mu}')
plt.plot(x, I_analytical, label="Analytical Solution", linestyle="dashed", color="black")
plt.xlabel('x')
plt.ylabel('I(x)')
plt.title('Finite Difference Solution of Radiative Transfer Equation')
plt.legend()
plt.grid()
plt.show()
