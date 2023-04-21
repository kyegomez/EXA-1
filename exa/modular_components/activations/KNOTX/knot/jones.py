import sympy as sp

def jones_polynomial_torus_knot(m, n):
    t = sp.symbols('t')
    numerator = t**((m-1) * (n-1)/2) * (1 - t ** (m + 1) - t**(n + 1) + t**(m+n))
    denominator = 1 - t **2
    return numerator / denominator



def knot_invariant(x):
    #convert the input value into a knot representation (m, n ) for a torus knot
    m, n = int(x), int(x + 1)

    #calculate the jones polynomial for the torus knot
    jones_poly = jones_polynomial_torus_knot(m, n)


    #eval the jones polynomial at a specific point
    knot_inv = jones_poly.subs('t', 2)

    return float(knot_inv)


#test the knot invariant function with sample input
input_value =  1.0
knot_inv_value = knot_invariant(input_value)
print(f"Knot invariant value after applying knot invariant function {knot_inv_value}")