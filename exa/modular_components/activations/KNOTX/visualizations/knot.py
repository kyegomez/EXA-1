import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp


def convert_to_knot_representation(x):
    m = int(np.ceil(x))
    n = m + 1
    return (m, n)


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


def visualize_knot_representation(x):
    device = torch.device('cpu')
    knot_reps = [convert_to_knot_representation(val) for val in x]

    # Calculate Lorenz system output for each knot representation
    lorenz_output = []
    for m, n in knot_reps:
        x0, y0, z0 = m, n, n + 1
        lorenz_output.append(lorenz_ode(x0, y0, z0)[-1])

    x_coords = x.cpu().numpy()
    y_coords = x.cpu().numpy() - 1
    z_coords = x.cpu().numpy() + 1

    # Shift coordinates by Lorenz system output
    y_coords -= np.array(lorenz_output)
    z_coords += np.array(lorenz_output)

    # Plot the knot representation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_coords, y_coords, z_coords)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


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


# Example usage
device = torch.device('cpu')
x = torch.tensor([[-10, -5, 0, 5], [1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device=device)
knotx_output = knotx(x, device)
visualize_knot_representation(knotx_output)