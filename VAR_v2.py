import numpy as np
import scipy.optimize

# input t*d matrix and parameter lambda_0, output d*d transition matrix
def VAR_v2(input,lambda_0):

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

    c = np.ones((1,2*dimension*dimension))

    B_ub = np.append(S1,-S1,axis=0)
    B_ub = B_ub.T
    B_ub = B_ub.reshape(-1,1)
    B_ub = B_ub + lambda_0

    A_ub0 = np.append( np.append(S,-S,axis=1), np.append(-S,S,axis=1),axis=0 )
    A_zero = np.zeros((2 * dimension, 2 * dimension))
    for i in range(dimension):
        for j in range(dimension):
            if j == 0:
                if i == 0:
                    A_ubi = A_ub0
                else:
                    A_ubi = A_zero
            else:  
                if j == i:
                    A_ubi = np.append(A_ubi,A_ub0,axis = 1)
                else:
                    A_ubi = np.append(A_ubi,A_zero,axis = 1)
        if i == 0:
            A_ub = A_ubi
        else:
            A_ub = np.append(A_ub,A_ubi,axis = 0)
    # A_ub = A_ub0 * np.identity(2*dimension*dimension)
    # A_ub = A_ub0
    # for i in range(dimension-1):
    #     A_ub = np.append(A_ub,A_ub0,axis = 1)

    res = scipy.optimize.linprog(c,A_ub,B_ub)

    x = res.x.reshape(dimension,2*dimension)

    res = np.hsplit(x,2)

    A_transition = (res[0]-res[1]).T
    return A_transition
