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
input = input.reshape(5,15)
input = input.T
curr_time = time.time()
print("get input:",curr_time-prev_time,"s")
prev_time = curr_time

print(input)

events_length = input.shape[0]
dimension = input.shape[1]



time_length = input.max() - input.min()
print(time_length)

m = 100
delta_h = time_length/m
Y = np.zeros((m,dimension))
input = (input - input.min())//delta_h
for idx,i in enumerate(input):
    for jdx,j in enumerate(i):
        if j == m:
            j = j - 1
        j = int(j)
        Y[j][jdx] = 1

print(input)
print(Y)
Y_bar = Y.sum(axis=0)/m
print(Y_bar)
Y_diff = Y - Y_bar
Y_diff_T = Y_diff.T
sum_Y = np.dot(Y_diff,Y_diff.T)

for kh in range(m-1): 
    k = kh + 1
    sum_lower_gamma = np.zeros(k)
    sum_upper_gamma = np.zeros((k,k))
    for th in range(m-k):
        t = th + k
        sum_lower_gamma += sum_Y[t,th:t]
        sum_upper_gamma += sum_Y[th:t,th:t]
    lower_gamma = sum_lower_gamma/(m-k)
    upper_gamma = sum_upper_gamma/(m-k)
    upper_gamma = np.mat(upper_gamma)
    theta = np.dot(lower_gamma,upper_gamma.I)
    print(k,":",theta)