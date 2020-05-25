# -*- coding: utf-8 -*-
import torch
import torch.nn
from layers.measurement import QMeasurement
from optimizer import *
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

time_per_epoch = []
dim = 200

#real part
a = torch.rand(16,dim,dim)

#imaginary part
b = torch.rand(16,dim,dim)


c = torch.rand(16,dim)
model = QMeasurement(dim)

criterion = torch.nn.MSELoss()
optimizer_1 = RMSprop_Unitary(model.parameters(),lr = 0.001, momentum = 0.99)   
#optimizer_2 = SGD_Unitary(model_2.parameters(),lr = 0.001,momentum = 0.9) 
#optimizer_3 = RMSprop_Unitary(model_3.parameters(),lr = 0.001,momentum = 0.9) 

model.train()
print('begin training!')
losses = []
for i in range(500):
    print('training epoch {}'.format(i))
    
    # Model 1
    optimizer_1.zero_grad()
    outputs = model([a,b])
  
    loss = criterion(outputs, c)
    loss.backward()  
    optimizer_1.step()
    print(loss.item())
    losses.append(loss.item())
#dims.append(dim)   
#_t = (time.time() - start_time)/20
#print(_t)
#time_per_epoch.append(_t)
    
#pickle.dump([dims, time_per_epoch], open('time_per_epoch.pkl','wb')) 