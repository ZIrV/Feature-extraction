import numpy as np
from ODE import *
from VAR import *

# input t*d matrix and parameter lambda_0, output d*d transition matrix
input = np.random.random((10,3)) 
lambda_0 = 0.05


print(VAR(input,lambda_0))

#input t-length series, the number of iterations and m (the slice of g(t)), output t*t probability matrix, parameter g(t) and mu
input = np.random.random(100)
input = np.sort(input)
iterations = 10
m = 100

print(ODE(input,iterations,m))