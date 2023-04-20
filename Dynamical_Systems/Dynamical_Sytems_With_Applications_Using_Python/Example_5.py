# from sympy import dsolve, Equality, Function, symbols

# t = symbols("t")
# x = symbols("x", cls=Function)

# sol = dsolve(eq=Equality(lhs=x(t).diff(t), 
#                          rhs= (9 - 12 * t - 5 * x(t)) / (5 * t + 2 * x(t) - 4), func=x(t))
# )
# print(sol)

# """
# [Eq(x(t), -5*t/2 - sqrt(C1 + t**2 - 4*t)/2 + 2), 
#  Eq(x(t), -5*t/2 + sqrt(C1 + t**2 - 4*t)/2 + 2)]
# """

# TODO - This is not working