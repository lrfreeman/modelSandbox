# Program 02b: The logistic equation - Verhulst’s equation. See Example 4.

# Dsolve - Solve ODEs symbolically - Solves any (supported) kind of ordinary differential equation
# and system of ordinary differential equations. Returns a list of solutions to the ODE
# symbols - Create symbol to namespace, accepts a range notation
# Function - Base class for mathematical functions
# Eq - Alias for Equality - An equal relation between two objects, so just an equation
# Example 4: Solve the following ODE using dsolve: dP/dt = P(Beta - delta*P)

from sympy import dsolve, Equality, Function, symbols

t = symbols("t")
P = symbols("p", cls=Function)
beta = symbols("B")
delta = symbols("d")

sol = dsolve(eq=Equality(lhs=P(t).diff(t), rhs= P(t) * (beta - delta * P(t))), func=P(t))
print(sol)
# result is a list of solutions to the ODE - 
# Eq(p(t), B/(d*(1 - exp(B*(C1 - t)))))
# This result can also be computed by hand 
# This is very close to the logisitc equation

# NOTE - When using dsolve the function such as P(t), x(t) must be written as such as not like P or x
