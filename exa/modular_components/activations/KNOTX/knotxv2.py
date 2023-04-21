import torch 
import numpy as np
from scipy.integrate import solve_ivp

#define the knot invarient function
def knot_invariant(x):
    #convert the input value x into a knot representation
    def knot_representation(x):
        return x * 2
    
    #calculate the knot invariant using a specific knot invariant algorithm [jones  polynomial]
    def jones_polynomial(knot_repr):
        return knot_repr ** 2
        
    knot_repr = knot_representation(x)
    knot_inv = jones_polynomial(knot_repr)

    return knot_inv


#define the lorenx system
def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt  = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]



#define the dynamical systems modeling function
def dynamical_systems_modeling(x, model="lorenz", params=None):
    if model == 'lorenz':
        #define the lorenz systems parameters
        if params is None:
            sigma, rho, beta = 10, 28, 8/3
        else:
            sigma, rho, beta = params

        #set initial state and time span
        initial_state = [x, x, x]
        t_span = (0, 1)

        #solve lorenz system
        sol = solve_ivp(lorenz_system, t_span, initial_state, args=(sigma, rho, beta), dense_output=True)


        #calculate the output value based on the applied dynamcal system model
        output_value = sol.sol(1)[0]

    return output_value



#define the KNOTX activation function
def knotx(x):
    knot_inv = knot_invariant(x)
    dyn_sys_mod = dynamical_systems_modeling(knot_inv)


    #create the hyper efficient version of gelu
    knotx_output =  0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))) * dyn_sys_mod)

    return knotx_output

#test the knotx activation function
input_values = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
output_values = torch.tensor([knotx(x) for x in input_values])
print(f"output values after applying knotx activation function {output_values}")

