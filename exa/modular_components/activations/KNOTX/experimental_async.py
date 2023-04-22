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
import torch.jit



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

# ===========================+>
#use a more efficient ode solver like the backward differentiation formula solver
# async def dynamical_systems_modeling(knot_representation):
#     sigma, rho, beta = 10, 28, 8/3
#     initial_state = list(knot_representation) + [0]  # Use knot_representation as the initial state
#     t_span = (0, 1)
#     # sol = solve_ivp(lorenz_system, t_span, initial_state, args=(sigma, rho, beta), dense_output=True)
#     sol = solve_ivp(lorenz_system, t_span, initial_state, args=(sigma, rho, beta), dense_output=True, method='BDF')

#     output_value = sol.sol(1)[0]  # Get the x value at t=1
#     return output_value
def dynamical_systems_modeling(knot_representation):
    sigma, rho, beta = 10, 28, 8/3
    initial_state = list(knot_representation) + [0]
    t_span = (0, 1)
    sol = solve_ivp(lorenz_system, t_span, initial_state, args=(sigma, rho, beta), dense_output=True, method='BDF')
    output_value = sol.sol(1)[0]
    return output_value



def vectorized_knot_invariant(x):
    m = np.ceil(x)
    n = m + 1
    return m * n



#torch built in vectorized function for faster computation
def fast_tanh(x):
    return torch.tanh(x)



# ===========================+>
#batch processing of multiple inputs 
async def parallel_lorenz_solver_batch(inital_states, batch_size=10):
    num_batches = len(initial_states) // batch_size
    results = []

    for i in range(num_batches):
        batch = initial_states[i * batch_size: (i + 1) * batch_size]
        batch_results = await parallel_lorenz_solver(batch)
        results.extend(batch_results)

    return results

# ===========================+>




# async def parallel_lorenz_solver(initial_states):
#     async with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = list(executor.map(dynamical_systems_modeling, initial_states))
#     return results

async def parallel_lorenz_solver(initial_states):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, dynamical_systems_modeling, state) for state in initial_states]
        results = await asyncio.gather(*tasks)
    return results


#v1
# def knotx(x, device):
#     x_flat = x.view(-1)
#     x_flat = x_flat.to(device)  # Move the tensor to the GPU if available

#     knot_inv = vectorized_knot_invariant(x_flat.detach().cpu().numpy())
#     lorenz_output = 1 + torch.tensor(fast_tanh(knot_inv**3), dtype=torch.float32, device=x.device).view_as(x_flat)

#     return x * lorenz_output.view_as(x)



#v2 --> converts a numpy array into a torch tensor
def knotx(x, device):
    x_flat = x.view(-1)
    x_flat = x_flat.to(device)  # Move the tensor to the GPU if available

    knot_inv = vectorized_knot_invariant(x_flat.detach().cpu().numpy())
    knot_inv_tensor = torch.tensor(knot_inv, dtype=torch.float32, device=x.device)  # Convert the NumPy array to a tensor
    lorenz_output = 1 + fast_tanh(knot_inv_tensor**3).view_as(x_flat)  # Use the tensor in fast_tanh

    return x * lorenz_output.view_as(x)


from torch.profiler import profile, record_function

def profile_knotx(x): 
    with profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("knotx"):
            result = knotx(x)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    return result

# def optimized_knotx(x):
#     x_flat = x.view(-1).to(device)
#     knot_inv = vectorized_knot_invariant(x_flat.detach().cpu().numpy())
#     lorenz_output = torch.tesnor(fast_tanh(knot_inv **3), dtype=torch.float32, device=x.device).view_as(x_flat)


#     #in place multiplication
#     x.mul_(lorenz_output.view_as(x))
#     return x

# def optimized_knotx(x):
#     x_flat = x.view(-1)
#     x_flat = x_flat.to(device)  # Move the tensor to the GPU if available

#     knot_inv = vectorized_knot_invariant(x_flat.detach().cpu().numpy())
#     lorenz_output = 1 + torch.tensor(fast_tanh(knot_inv**3), dtype=torch.float32, device=x.device).view_as(x_flat)

#     return x * lorenz_output.view_as(x)


#optimized knotx for torch jit
# @torch.jit.script
# def optimized_knotx(x: torch.Tensor, device: torch.device) -> torch.Tensor:
#     x_flat = x.view(-1)
#     x_flat = x_flat.to(device)  # Move the tensor to the GPU if available

#     knot_inv = vectorized_knot_invariant(x_flat.detach().cpu().numpy())
#     lorenz_output = 1 + torch.tensor(fast_tanh(knot_inv**3), dtype=torch.float32, device=x.device).view_as(x_flat)

#     return x * lorenz_output.view_as(x)


#v2
# def optimized_knotx(x, device):
#     x_flat = x.view(-1)
#     x_flat = x_flat.to(device)  # Move the tensor to the GPU if available

#     knot_inv = vectorized_knot_invariant(x_flat.detach().cpu().numpy())
#     lorenz_output = 1 + torch.tensor(fast_tanh(knot_inv**3), dtype=torch.float32, device=x.device).view_as(x_flat)

#     return x * lorenz_output.view_as(x)


#v3 --> transforming numpy array into torch tensor
def optimized_knotx(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x_flat = x.view(-1)
    x_flat = x_flat.to(device)  # Move the tensor to the GPU if available

    knot_inv = vectorized_knot_invariant(x_flat.detach().cpu().numpy())
    knot_inv_tensor = torch.tensor(knot_inv, dtype=torch.float32, device=x.device)  # Convert the NumPy array to a tensor
    lorenz_output = 1 + fast_tanh(knot_inv_tensor**3).view_as(x_flat)  # Use the tensor in fast_tanh

    return x * lorenz_output.view_as(x)



initial_states = [convert_to_knot_representation(x) for x in [0.5, 1.0, 1.5]]
results = asyncio.run(parallel_lorenz_solver(initial_states))
print(results)

import timeit
import psutil
import os

x = torch.randn(1000, device=device)  # Create a random tensor of shape (1000,) for testing

# Update the measure_time_and_memory function to pass the device as an argument
def measure_time_and_memory(func, x, device, num_runs=100):
    start_time = timeit.default_timer()
    start_memory = psutil.Process(os.getpid()).memory_info().rss

    for _ in range(num_runs):
        result = func(x.clone(), device)  # Pass the device as an argument

    end_time = timeit.default_timer()
    end_memory = psutil.Process(os.getpid()).memory_info().rss

    time_elapsed = (end_time - start_time) / num_runs
    memory_used = end_memory - start_memory

    return time_elapsed, memory_used

# ...

time_elapsed, memory_used = measure_time_and_memory(knotx, x, device)
print(f"Original function: Time elapsed = {time_elapsed:.6f} s, Memory used = {memory_used / 1024} KiB")

time_elapsed, memory_used = measure_time_and_memory(optimized_knotx, x, device)
print(f"Optimized function: Time elapsed = {time_elapsed:.6f} s, Memory used = {memory_used / 1024} KiB")