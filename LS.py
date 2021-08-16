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
