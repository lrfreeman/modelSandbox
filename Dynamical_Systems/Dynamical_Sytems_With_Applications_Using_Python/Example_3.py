# Program 02a: A simple separable ODE. See Example 1.
# Dsolve - Solve ODEs symbolically - Solves any (supported) kind of ordinary differential equation
# and system of ordinary differential equations. Returns a list of solutions to the ODE
# symbols - Create symbol to namespace, accepts a range notation
# Function - Base class for mathematical functions
# Eq - Alias for Equality - An equal relation between two objects, so just an equation
# Example 3: Solve the following seperable ODE using dsolve: dx/dt = t/x**2

from sympy import dsolve, Equality, Function, symbols

t = symbols("t")
x = symbols("x", cls=Function)
sol = dsolve(eq=Equality(lhs=x(t).diff(t), rhs= t / x(t)**2), func=x(t))
print(sol)
# result is a list of solutions to the ODE - 
# [Eq(x(t), (C1 + 3*t**2/2)**(1/3)), - This one is very close to the result produced by hand in the text book 
# Eq(x(t), (-1 - sqrt(3)*I)*(C1 + 3*t**2/2)**(1/3)/2), 
# Eq(x(t), (-1 + sqrt(3)*I)*(C1 + 3*t**2/2)**(1/3)/2)]
# This result can also be computed by hand 
