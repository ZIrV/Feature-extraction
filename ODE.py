import numpy as np

#input t-length series, the number of iterations and m (the slice of g(t)), output t*t probability matrix, parameter g(t) and mu
def ODE(input, iterations, m):

    events_length = input.shape[0]

    start = 100000
    end = 0
    for i in input:
        if i < start:
            start = i
        if i > end:
            end = i
    time_length = end - start
    delta_t = time_length/m
    g = np.zeros(m)
    mu = 0
    probability = np.zeros((events_length, events_length))

    for j in range(events_length):
        for i in range(events_length):
            if i > j:
                probability[i][j] = 0
            else:
                probability[i][j] = 1/(j+1)

    prev = np.array(probability,copy=True)
    for k in range(iterations):
        
        g = np.zeros(m)
        mu = 0
        for idx,i in enumerate(input):

            mu += probability[idx][idx]
            for jdx in range(events_length-idx-1):
                jdx = jdx + idx + 1

                itv = int((input[jdx]-i)//delta_t)
                if itv == m:
                    itv -= 1
                g[itv] += probability[idx][jdx]
        
        g /= delta_t
        mu /= events_length


        for jdx,j in enumerate(input): 
            temp_sum = mu
            for idx in range(jdx):
                
                itv = int((j-input[idx])//delta_t)
                if itv == m:
                    itv -= 1
                temp_sum += g[itv]

            probability[jdx][jdx] = mu / temp_sum
            for idx in range(jdx):
                itv = int((j-input[idx])//delta_t)
                if itv == m:
                    itv -= 1
                probability[idx][jdx] = g[itv] / temp_sum

        # print(np.maximum(probability-prev,prev-probability).sum())
        # prev = np.array(probability,copy=True)
        # print(g)
        # print(mu)
    
    return probability,g,mu