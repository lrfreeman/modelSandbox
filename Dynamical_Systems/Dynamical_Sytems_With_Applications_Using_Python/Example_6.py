from sympy import dsolve, Equality, Function, symbols

t = symbols("t")
x = symbols("x", cls=Function)

sol = dsolve(eq=Equality(lhs=x(t).diff(t), rhs=(t - x(t)) / (t + x(t)), func=x(t)))

print(sol)

# The solution [Eq(x(t), -t - sqrt(C1 + 2*t**2)), Eq(x(t), -t + sqrt(C1 + 2*t**2))] is equivilent to the book

import numpy as np
import matplotlib.pyplot as plt


# Define the function for the implicit equation x^2 + 2tx - t^2 = C provided by the book
def implicit_eq(x, t, C):
    return x**2 + 2 * t * x - t**2 - C


# Set up the plot
plt.figure(figsize=(10, 6))

# Choose a range of t and x values
t = np.linspace(-4, 4, 1000)
x = np.linspace(-4, 4, 1000)

# Create a grid for t and x values
T, X = np.meshgrid(t, x)

# Choose a range of C values and plot the corresponding solution curves
C_values = np.arange(-8, 9, 2)
for C in C_values:
    plt.contour(T, X, implicit_eq(X, T, C), levels=[0], colors="blue")

# Customize the plot
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Solution curves for x^2 + 2tx - t^2 = C")
plt.grid()
plt.show()
