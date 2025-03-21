import numpy as np
import matplotlib.pyplot as plt

def load_solution(mu, ke):
    # Construct filename based on parameters
    filename = f'radiation_solution_mu_{mu:.2f}_ke_{ke:.2f}.csv'
    
    try:
        # Load the solution from CSV
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        return data[:, 0], data[:, 1]  # Return x and I values
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
        print("Make sure to run 1d_inf_wall.py with these parameters first.")
        return None, None

def load_pinn_solution(ke):
    # Load PINN solution
    filename = f'pinn_solution_ke_{ke:.1f}.csv'
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        return data[:, 0], data[:, 1]
    except FileNotFoundError:
        print(f"Error: Could not find PINN solution file {filename}")
        return None, None

# Parameters for the solutions
solutions = [
    {'mu': 0.5, 'ke': 0.1},
    {'mu': 0.5, 'ke': 1.0},
    {'mu': 0.5, 'ke': 10.0}
]

# Create a single plot for comparison
plt.figure(figsize=(10, 6))

# Load and plot each analytical solution and corresponding PINN solution
for params in solutions:
    mu = params['mu']
    ke = params['ke']
    
    # Load and plot analytical solution
    x, I = load_solution(mu, ke)
    if x is not None and I is not None:
        plt.plot(x, I, label=f'Finite Difference (κ={ke:.2f})', linewidth=4)
    
    # Load and plot corresponding PINN solution
    x_pinn, I_pinn = load_pinn_solution(ke)
    if x_pinn is not None and I_pinn is not None:
        plt.plot(x_pinn, I_pinn, '--', label=f'PINN (κ={ke:.1f})', linewidth=4)

plt.xlabel('x')
plt.ylabel('I(x)')
plt.title('Comparison of 1D-RTE Solutions: Analytical vs PINN')
plt.legend()
plt.grid(True)
plt.savefig("1d_rte_comparison.pdf", dpi=300, bbox_inches='tight')
plt.show() 