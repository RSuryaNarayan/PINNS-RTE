import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the Physics-Informed Neural Network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1), nn.Tanh()
        )

    def forward(self, x, ke):
        inputs = torch.cat((x, ke), dim=1)
        return self.net(inputs)

def loss_function(model, x, ke, mu, source):
    x.requires_grad = True
    ke.requires_grad = False  # We treat ke as a parameter, not a learnable variable
    
    I_pred = model(x, ke)
    
    # Compute dI/dx using autograd
    dI_dx = torch.autograd.grad(I_pred, x, grad_outputs=torch.ones_like(I_pred),
                                create_graph=True)[0]
    
    # Residual loss (enforcing the ODE)
    residual = mu * dI_dx + ke * I_pred - source
    loss_residual = torch.mean(residual**2)
    
    # Extract boundary values from I_pred
    I_left = I_pred[0]   # I(0, ke)
    # I_right = I_pred[-1] # I(1, ke)
    
    # Boundary condition loss
    BC_left = torch.mean((I_left - (1/ke) * torch.exp(-c**2 / alpha**2))**2)
    # BC_right = torch.mean((I_right - (1/ke) * torch.exp(-((1-c)**2) / alpha**2))**2)

    return loss_residual + BC_left

# Training the PINN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
Nx = 4000
x_train = torch.linspace(0, 1, Nx).view(-1, 1).to(device)

ke_values = torch.linspace(10.0, 10.0, 1).view(-1, 1).to(device)  # Training on a range of ke values
mu = torch.tensor(0.5).to(device)  # Fixed mu
c = torch.tensor(0.5).to(device)  # Convert to tensor
alpha = torch.tensor(0.02).to(device)  # Convert to tensor

# Define the source term
source_train = torch.exp(-((x_train - c) ** 2) / alpha**2).to(device)

# Initialize model
model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Sample a random ke for each training step
    ke_train = ke_values[torch.randint(0, len(ke_values), (1,))].expand(Nx, 1).to(device)  # Fixed ke for this batch
    
    loss = loss_function(model, x_train, ke_train, mu, source_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Plot results for a fixed ke
ke_fixed = torch.tensor([[10.0]]).to(device)  # Example for ke = 1.0

with torch.no_grad():
    I_pred = model(x_train, ke_fixed.expand(Nx, 1)).cpu().numpy()  # Keep ke constant for all x

# Save solution to CSV file
solution_data = np.column_stack((x_train.detach().cpu().numpy(), I_pred))
np.savetxt('pinn_solution_ke_10.0.csv', solution_data, delimiter=',', header='x,intensity', comments='')

# Plot results
plt.plot(x_train.detach().cpu().numpy(), I_pred, label="PINN Prediction (ke=1.0)")
plt.xlabel("x")
plt.ylabel("I(x)")
plt.legend()
plt.savefig("1d_infWall_pinn.pdf", dpi=300, bbox_inches='tight')
plt.show()
