import numpy as np
import pickle
import time
from ODE import *
from VAR import *
from VAR_v2 import *

import warnings
warnings.filterwarnings("ignore")
# input t*d matrix and parameter lambda_0, output d*d transition matrix
prev_time = time.time()
f=open('../../data_so-20210704T114021Z-001/data_so/fold1/train.pkl','rb')#文件所在路径
pkl=pickle.load(f,encoding='latin1')#读取pkl内容

data = pkl["train"]

curr_time = time.time()
print("read file:",curr_time-prev_time,"s")
prev_time = curr_time

input = []
for j in data[0:10]:
    input_row = []
    if len(j) < 30:
        for i in range(30 - len(j)):
            input_row.append(0)
        for i in j:
            input_row.append(i['time_since_start'])
    else:
        for i in j[0:30]:
            input_row.append(i['time_since_start'])
    input = np.append(input,input_row)
input = input.reshape(30,10)
curr_time = time.time()
print("get input:",curr_time-prev_time,"s")
prev_time = curr_time

lambda_0 = 0.05

print(VAR(input,lambda_0))

curr_time = time.time()
print("get input:",curr_time-prev_time,"s")
prev_time = curr_time

#input t-length series, the number of iterations and m (the slice of g(t)), output t*t probability matrix, parameter g(t) and mu
# input = np.random.random(100)
# input = np.sort(input)
# iterations = 10
# m = 100

# print(ODE(input,iterations,m))