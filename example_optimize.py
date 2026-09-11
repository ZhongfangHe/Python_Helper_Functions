# Examples of optimization


import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint


def rosen_with_args(x, a, b):
    """The Rosenbrock function with additional arguments"""
    return sum(a*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0) + b
    

# Unconstrained 
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res1 = minimize(rosen_with_args, x0, args=(0.5, 1.), method='bfgs',
               options={'disp': True})


# Linear constraints, bounds
bounds = Bounds([0, -0.5], [1.0, 2.0])
linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])
x0 = np.array([0.5, 0])
res2 = minimize(rosen_with_args, x0, args=(0.5, 1.), 
                bounds=bounds, constraints=linear_constraint,
                method='trust-constr',options={'disp': True})