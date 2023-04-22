import torch
import numpy as np
from scipy.integrate import solve_ivp
import asyncio
import concurrent.futures

# Set the default device to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lorenz_ode(x0, y0, z0, sigma=10, rho=28, beta=8/3, dt=0.01, steps=1000):
    x, y, z = x0, y0, z0
    for _ in range(steps):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x, y, z = x + dx, y + dy, z + dz
    return x, y, z


def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def convert_to_knot_representation(x):
    m = int(np.ceil(x))
    n = m + 1
    return (m, n)

def vectorized_knot_invariant(x):
    m = np.ceil(x)
    n = m + 1
    return m * n

def fast_tanh(x):
    return torch.tanh(x)

def dynamical_systems_modeling(knot_representation):
    sigma, rho, beta = 10, 28, 8/3
    initial_state = list(knot_representation) + [0]
    t_span = (0, 1)
    sol = solve_ivp(lorenz_system, t_span, initial_state, args=(sigma, rho, beta), dense_output=True, method='BDF')
    output_value = sol.sol(1)[0]
    return output_value

async def parallel_lorenz_solver(initial_states):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, dynamical_systems_modeling, state) for state in initial_states]
        results = await asyncio.gather(*tasks)
    return results

# def knotx(x, device):
#     x_flat = x.view(-1)
#     x_flat = x_flat.to(device)

#     knot_representation = np.array([convert_to_knot_representation(val.item()) for val in x_flat])
#     lorenz_output = asyncio.run(parallel_lorenz_solver(knot_representation))
#     lorenz_output = torch.tensor(lorenz_output, dtype=torch.float32, device=x.device).view_as(x_flat)

#     return x * (1 + lorenz_output)


def knotx(x, device):
    output = torch.empty_like(x)
    for i in range(x.shape[0]):
        x0, y0, z0 = x[i], x[i] + 1, x[i] + 2
        output[i] = lorenz_ode(x0, y0, z0)[-1]
    return output


from torch.profiler import profile, record_function

def profile_knotx(x): 
    with profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("knotx"):
            result = knotx(x)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    return result



# def optimized_knotx(x: torch.Tensor, device: torch.device) -> torch.Tensor:
#     x_flat = x.view(-1)
#     x_flat = x_flat.to(device)

#     knot_inv = vectorized_knot_invariant(x_flat.detach().cpu().numpy())
#     knot_inv_tensor = torch.tensor(knot_inv, dtype=torch.float32, device=x.device)
#     lorenz_output = 1 + fast_tanh(knot_inv_tensor**3).view_as(x_flat)

#     return x * lorenz_output.view_as(x)

def optimized_knotx(x, device):
    x0, y0, z0 = x, x + 1, x + 2
    x, y, z = lorenz_ode(x0, y0, z0)
    return z

import timeit
import psutil
import os

x = torch.randn(1000, device=device)  # Create a random tensor of shape (1000,) for testing


def measure_time_and_memory(func, x, device, num_runs=100):
    start_time = timeit.default_timer()
    start_memory = psutil.Process(os.getpid()).memory_info().rss

    for _ in range(num_runs):
        result = func(x.clone(), device)

    end_time = timeit.default_timer()
    end_memory = psutil.Process(os.getpid()).memory_info().rss

    time_elapsed = (end_time - start_time) / num_runs
    memory_used = end_memory - start_memory

    return time_elapsed, memory_used

initial_states = [convert_to_knot_representation(x) for x in [0.5, 1.0, 1.5]]
results = asyncio.run(parallel_lorenz_solver(initial_states))
print(results)



time_elapsed, memory_used = measure_time_and_memory(knotx, x, device)
print(f"Original function: Time elapsed = {time_elapsed:.6f} s, Memory used = {memory_used / 1024} KiB")

time_elapsed, memory_used = measure_time_and_memory(optimized_knotx, x, device)
print(f"Optimized function: Time elapsed = {time_elapsed:.6f} s, Memory used = {memory_used / 1024} KiB")


# Check if the optimized function produces the same output as the original function
x_test = torch.randn(1000, device=device)
original_output = knotx(x_test.clone(), device)
optimized_output = optimized_knotx(x_test.clone(), device)

assert torch.allclose(original_output, optimized_output), "The outputs of the original and optimized functions do not match"

# Profile the optimized function
profiled_output = profile_knotx(x_test.clone())
assert torch.allclose(original_output, profiled_output), "The outputs of the original and profiled functions do not match"

# Save the optimized function as a TorchScript module
optimized_knotx_script = torch.jit.script(optimized_knotx)
torch.jit.save(optimized_knotx_script, "optimized_knotx.pt")

# Load the saved TorchScript module and test it
loaded_optimized_knotx = torch.jit.load("optimized_knotx.pt")
loaded_output = loaded_optimized_knotx(x_test.clone(), device)
assert torch.allclose(original_output, loaded_output), "The outputs of the original and loaded functions do not match"

print("All tests passed!")