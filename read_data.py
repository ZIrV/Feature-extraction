import pickle
import time
from VAR import *

import warnings
warnings.filterwarnings("ignore")

prev_time = time.time()
f=open('../../data_so-20210704T114021Z-001/data_so/fold1/train.pkl','rb')#文件所在路径
pkl=pickle.load(f,encoding='latin1')#读取pkl内容

data = pkl["train"]

curr_time = time.time()
print("read file:",curr_time-prev_time,"s")
prev_time = curr_time

input = []
for j in data[0:5]:
    input_row = []
    if len(j) < 15:
        for i in range(15 - len(j)):
            input_row.append(0)
        for i in j:
            input_row.append(i['time_since_start'])
    else:
        for i in j[0:15]:
            input_row.append(i['time_since_start'])
    input = np.append(input,input_row)
input = input.reshape(15,5)
curr_time = time.time()
print("get input:",curr_time-prev_time,"s")
prev_time = curr_time

lambda_0 = 0.05

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

curr_time = time.time()
print("calculate paraments:",curr_time-prev_time,"s")
prev_time = curr_time


c = np.ones((1,2*dimension))

A_ub = np.append( np.append(S,-S,axis=1), np.append(-S,S,axis=1),axis=0 )

A_transition = np.zeros((dimension,dimension))
for j in range(dimension):

    B_ub = np.append( S1[:,j].reshape(dimension,1) + lambda_0 * np.ones((dimension,1)), - S1[:,j].reshape(dimension,1) + lambda_0*np.ones((dimension,1)),axis=0 )

    res = scipy.optimize.linprog(c,A_ub,B_ub)

    x = res.x.reshape(2,dimension)

    x = x[0]-x[1]

    A_transition[:,j] = x

curr_time = time.time()
print("get result:",curr_time-prev_time,"s")
prev_time = curr_time

print(A_transition)