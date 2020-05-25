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
#dims =[]
#for i in range(50):

#dim = (1+i)*10
#print(dim)
a = torch.rand(16,dim,dim)
b = torch.rand(16,dim,dim)
c = torch.rand(16,dim)
model_1 = QMeasurement(dim)
model_2 = QMeasurement(dim)
model_3 = QMeasurement(dim)

criterion = torch.nn.MSELoss()
optimizer_1 = Vanilla_Unitary(model_1.parameters(),lr = 0.001)   
optimizer_2 = SGD_Unitary(model_2.parameters(),lr = 0.001,momentum = 0.9) 
optimizer_3 = RMSprop_Unitary(model_3.parameters(),lr = 0.001,momentum = 0.9) 

model_1.train()
model_2.train()
model_3.train()
print('begin training!')
start_time = time.time()
losses_1 = []
losses_2 = []
losses_3 = []
for i in range(500):
    print('training epoch {}'.format(i))
    
    # Model 1
    optimizer_1.zero_grad()
    outputs_1 = model_1([a,b])
  
    loss = criterion(outputs_1, c)
    loss.backward()  
    optimizer_1.step()
    z = model_1.kernel[:,:,0].cpu().detach().numpy() +1j* model_1.kernel[:,:,1].cpu().detach().numpy()
    real_part_loss = (np.matmul(z.real, z.real.transpose()) + np.matmul(z.imag, z.imag.transpose()) - np.eye(dim))
    imag_part_loss = np.matmul(z.imag, z.real.transpose()) - np.matmul(z.real, z.imag.transpose())
    s = (real_part_loss**2 + imag_part_loss **2 )**0.5
    losses_1.append(s.max())
    
    optimizer_2.zero_grad()
    outputs_2 = model_2([a,b])
  
    loss_2 = criterion(outputs_2, c)
    loss_2.backward()  
    optimizer_2.step()
    z = model_2.kernel[:,:,0].cpu().detach().numpy() +1j* model_2.kernel[:,:,1].cpu().detach().numpy()
    real_part_loss = (np.matmul(z.real, z.real.transpose()) + np.matmul(z.imag, z.imag.transpose()) - np.eye(dim))
    imag_part_loss = np.matmul(z.imag, z.real.transpose()) - np.matmul(z.real, z.imag.transpose())
    s = (real_part_loss**2 + imag_part_loss **2 )**0.5
    losses_2.append(s.max())
    
    optimizer_3.zero_grad()
    outputs_3 = model_3([a,b])
    loss_3 = criterion(outputs_3, c)
    loss_3.backward()  
    optimizer_3.step()
    z = model_3.kernel[:,:,0].cpu().detach().numpy() +1j* model_3.kernel[:,:,1].cpu().detach().numpy()
    real_part_loss = (np.matmul(z.real, z.real.transpose()) + np.matmul(z.imag, z.imag.transpose()) - np.eye(dim))
    imag_part_loss = np.matmul(z.imag, z.real.transpose()) - np.matmul(z.real, z.imag.transpose())
    s = (real_part_loss**2 + imag_part_loss **2 )**0.5
    losses_3.append(s.max())
    


pickle.dump([losses_1, losses_2, losses_3], open('losses.pkl','wb'))
#dims.append(dim)   
#_t = (time.time() - start_time)/20
#print(_t)
#time_per_epoch.append(_t)
    
#pickle.dump([dims, time_per_epoch], open('time_per_epoch.pkl','wb')) 

