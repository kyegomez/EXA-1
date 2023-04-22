import torch
import numpy as np
from scipy.integrate import solve_ivp
import torch.optim
import sympy as sp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import concurrent.futures
import asyncio


# Set the default device to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def jones_polynomial_torus_knot(m, n):
    t = sp.symbols('t')
    numerator = t**((m-1) * (n-1)/2) * (1 - t ** (m + 1) - t**(n + 1) + t**(m+n))
    denominator = 1 - t **2
    return numerator / denominator

def convert_to_knot_representation(x):
    # Convert x to a suitable knot representation, for example, a torus knot (m, n)
    m = int(np.ceil(x))
    n = m + 1
    return (m, n)

def knot_invariant(x):
    knot_representation = convert_to_knot_representation(x)
    m, n = knot_representation
    return m * n

async def dynamical_systems_modeling(knot_representation):
    sigma, rho, beta = 10, 28, 8/3
    initial_state = list(knot_representation) + [0]  # Use knot_representation as the initial state
    t_span = (0, 1)
    sol = solve_ivp(lorenz_system, t_span, initial_state, args=(sigma, rho, beta), dense_output=True)
    output_value = sol.sol(1)[0]  # Get the x value at t=1
    return output_value

def vectorized_knot_invariant(x):
    m = np.ceil(x)
    n = m + 1
    return m * n

def fast_tanh(x):
    return x / (1 + np.abs(x))

async def parallel_lorenz_solver(initial_states):
    async with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(dynamical_systems_modeling, initial_states))
    return results

def knotx(x):
    x_flat = x.view(-1)
    x_flat = x_flat.to(device)  # Move the tensor to the GPU if available

    knot_inv = vectorized_knot_invariant(x_flat.detach().cpu().numpy())
    lorenz_output = 1 + torch.tensor(fast_tanh(knot_inv**3), dtype=torch.float32, device=x.device).view_as(x_flat)

    return x * lorenz_output.view_as(x)


from torch.profiler import profile, record_function

def profile_knotx(x): 
    with profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("knotx"):
            result = knotx(x)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    return result

def optimized_knotx(x):
    x_flat = x.view(-1).to(device)
    knot_inv = vectorized_knot_invariant(x_flat.detach().cpu().numpy())
    lorenz_output = torch.tesnor(fast_tanh(knot_inv **3), dtype=torch.float32, device=x.device).view_as(x_flat)


    #in place multiplication
    x.mul_(lorenz_output.view_as(x))
    return x


