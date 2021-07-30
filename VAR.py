import numpy as np
import scipy.optimize

# input t*d matrix and parameter lambda_0, output d*d transition matrix
def VAR(input,lambda_0):

    events_length = input.shape[0]
    dimension = input.shape[1]

    sum = np.zeros((dimension, dimension))
    sum1 = np.zeros((dimension, dimension))
    prev_transposed = np.zeros((dimension,1))

    for i in input:
        i = i.reshape(1,dimension)
        i_transposed = i.reshape(dimension,1)

        sum += np.dot(i_transposed,i)
        sum1 += np.dot(prev_transposed,i)
        prev_transposed = np.array(i_transposed,copy=True)

    S = sum/events_length
    S1 = sum1/(events_length-1)

    A_transition = np.zeros((dimension,dimension))
    for j in range(dimension):

        lambda_0 = lambda_0

        c = np.ones((1,2*dimension))

        A_ub = np.append( np.append(S,-S,axis=1), np.append(-S,S,axis=1),axis=0 )

        B_ub = np.append( S1[:,j].reshape(dimension,1) + lambda_0*np.ones((dimension,1)), - S1[:,j].reshape(dimension,1) + lambda_0*np.ones((dimension,1)),axis=0 )

        res = scipy.optimize.linprog(c,A_ub,B_ub)

        x = res.x.reshape(2,dimension)
        x = x[0]-x[1]

        A_transition[:,j] = x

    return A_transition
