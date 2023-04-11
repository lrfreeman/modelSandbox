# Program 02a: A simple separable ODE. See Example 1.
# Dsolve - Solve ODEs symbolically - Solves any (supported) kind of ordinary differential equation
# and system of ordinary differential equations. Returns a list of solutions to the ODE
# symbols - Create symbol to namespace, accepts a range notation
# Function - Base class for mathematical functions
# Eq - Alias for Equality - An equal relation between two objects.
# Example 2: Solve the following seperable ODE using dsolve: dx/dt = -t/x

from sympy import dsolve, Equality, Function, symbols

t = symbols("t")
x = symbols("x", cls=Function)
sol = dsolve(eq=Equality(lhs=x(t).diff(t), rhs=-t / x(t)), func=x(t))
print(sol)
# result is a list of solutions to the ODE - [Eq(x(t), -sqrt(C1 - t**2)), Eq(x(t), sqrt(C1 - t**2))]
# This result can also be computed by hand 
