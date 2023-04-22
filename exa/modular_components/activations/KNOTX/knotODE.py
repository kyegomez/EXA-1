import numpy as np
import torch
from scipy.integrate import solve_ivp

def convert_to_knot_representation(x):
    m = int(np.ceil(x))
    n = m + 1
    return (m, n)

def lorenz_system(t, y, sigma=10, rho=28, beta=8/3):
    x, y, z = y
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def lorenz_ode(x0, y0, z0, t_span=(0, 10), t_eval=None):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)

    sol = solve_ivp(
        lorenz_system,
        t_span,
        (x0, y0, z0),
        t_eval=t_eval,
        method="RK45",
        args=(10, 28, 8/3)
    )

    return sol.y[:, -1]

def knotx(x, device):
    x_flat = x.view(-1)
    x_flat = x_flat.to(device)

    knot_representation = np.array([convert_to_knot_representation(val.item()) for val in x_flat])

    lorenz_output = []
    for m, n in knot_representation:
        x0, y0, z0 = m, n, n + 1
        lorenz_output.append(lorenz_ode(x0, y0, z0)[-1])

    lorenz_output = torch.tensor(lorenz_output, dtype=torch.float32, device=x.device).view_as(x_flat)

    return x * (1 + lorenz_output)


# def knotx(x, device):
#     output = torch.empty_like(x)
#     for i in range(x.shape[0]):
#         x0, y0, z0 = x[i], x[i] + 1, x[i] + 2
#         output[i] = lorenz_ode(x0, y0, z0)[-1]
#     return output


# def test_knotx():
#     device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#     #create a random input tensor with the shape (batch size 1)
#     batch_size = 5
#     x = torch.rand(batch_size, 1) * 10


#     #run the knotx function on the input token
#     output = knotx(x, device)

#     #check if the output tensor has the same shape as the input tensor

#     assert output.shape == x.shape, f"output shape {output.shape} does not match input shape {x.shape}"

#     #check if the output values are updated as expected
#     for i in range(batch_size):
#         x_val = x[i].item()
#         expected_output = x_val * (1 + lorenz_ode(*convert_to_knot_representation(x_val), x_val + 1, x_val + 2)[-1])
#         assert np.isclose(output[i].item(), expected_output, rtol=1e-5), f"Output value {output[i].item()} does not match expected value {expected.output}"

#     print("knotx test passed")

# test_knotx()



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate initial conditions for the Lorenz system
x0, y0, z0 = 1, 1, 1

# Solve the Lorenz system and get the output trajectory
t_span = (0, 100)
sol = solve_ivp(lorenz_system, t_span, (x0, y0, z0), t_eval=np.linspace(*t_span, 10000))
x, y, z = sol.y

# Create a 3D plot of the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
plt.show()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#====================> knotx visulization
# Define the range of input values for x
x_range = np.linspace(-10, 10, 100)

# Generate the knotx output for each input value
knotx_output = knotx(torch.tensor(x_range, dtype=torch.float32, device=device), device)

# Use the knotx output to generate x, y, and z coordinates for the Lorenz system
x_coords = knotx_output.detach().cpu().numpy()
y_coords = knotx_output.detach().cpu().numpy() - 1
z_coords = knotx_output.detach().cpu().numpy() + 1

# Initialize a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the Lorenz system trajectory
ax.plot(x_coords, y_coords, z_coords)

# Set the axis labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Show the plot
plt.show()