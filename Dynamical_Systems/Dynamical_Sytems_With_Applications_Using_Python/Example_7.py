## Ploted solution curves myself and there is no comparison to book so not sure if correct
## Two solutions, one is analytically and the other is to hint a numerical solution i.e power series 

from sympy import dsolve, Equality, Function, symbols

t = symbols("t")
x = symbols("x", cls=Function)

sol = dsolve(eq=Equality(lhs=x(t).diff(t), rhs= t**3 - t * x(t), func=x(t)))

print(sol)

# The solution Eq(x(t), C1*exp(-t**2/2) + t**2 - 2) is equivilent to the book

# Series solution instead

# Program 02c : Power series solution first order ODE. # See Example 7. 
from sympy import dsolve, Function, pprint 
from sympy.abc import t 
x = Function("x") 
ODE1 = x(t).diff(t) + t*x(t) - t**3 
pprint(dsolve(ODE1, hint="1st_power_series", n=8, ics={x(0):1}))

## Series solution seems to match the book though not familar with the syntax

# Plotting ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# # Define the function for the implicit equation x(t) = -2 + t**2 + 3e^-t**2/2 provided by the book
def implicit_eq(x, t, C):
    return -x - 2 + t**2 + 3 * np.exp(-t**2 / 2) - C

# # Set up the plot
plt.figure(figsize=(10, 6))

# # Choose a range of t and x values
t = np.linspace(-4, 4, 1000)
x = np.linspace(-4, 4, 1000)

# # Create a grid for t and x values
T, X = np.meshgrid(t, x)

# # Choose a range of C values and plot the corresponding solution curves
C_values = np.arange(-8, 9, 2)
for C in C_values:
    plt.contour(T, X, implicit_eq(X, T, C), levels=[0], colors="blue")

# # Customize the plot
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Solution curves for x(t) = -2 + t**2 + 3e^-t**2/2")
plt.grid()
plt.show()
