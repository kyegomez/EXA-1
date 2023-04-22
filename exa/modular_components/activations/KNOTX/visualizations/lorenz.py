import jax
import jax.numpy as jnp

@jit
def lorenz(sigma, beta, rho, X, t):
    x, y, z = X

    xdot = sigma * (y - x)
    ydot = X * (rho - z) - y
    zdot = x * y - beta * z


    return jnp.array([xdot, ydot, zdot])

#since th params are fixed we use a partial to create aa new function that does not ask them
g = partial(lorenz, 10,8 / 3, 28)

g = jit(g)

#inital condition 
x_0 = jnp.ones(3)

#time intervals
t_vals = jnp.linspace(0., 450., 45000)

#integrate the function to get the data
sol = odeint(g, x_0, t_vals)

X = sol[500:]