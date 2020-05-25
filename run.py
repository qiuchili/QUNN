# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:44:48 2019

@author: qiuchi
"""
import torch
import random
from params import Params
import os
import numpy as np
import models
from models.qnn import QNN
from models.qattn import QAttN
import argparse
import pandas as pd
import pickle
from optimizer import *
from torch.nn import NLLLoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time


def set_seed(params):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(params.seed)
    os.environ['PYTHONHASHSEED'] = str(params.seed)
    np.random.seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)
    else:
        torch.manual_seed(params.seed)
    
def run(params):   
    model = QAttN(params)
    num_samples = 1000
    time_stamps = 100
    inputs = torch.rand(num_samples, time_stamps, params.input_dim)
    targets = torch.rand(num_samples, time_stamps, params.output_dim)

    train_dataset = TensorDataset(inputs,targets)
    data_loader = DataLoader(train_dataset,batch_size = params.batch_size, shuffle = True)

    model = model.to(params.device)


    criterion = NLLLoss()

    if hasattr(model,'get_params'):
        unitary_params, remaining_params = model.get_params()
    else:
        remaining_params = model.parameters()
        unitary_params = []
        
    if len(unitary_params)>0:
        # Could be replaced with SGD_Unitary or Vanilla_Unitary
        unitary_optimizer = RMSprop_Unitary(unitary_params,lr = params.unitary_lr)

    optimizer = torch.optim.RMSprop(remaining_params,lr = params.lr)  
    for i in range(params.epochs):
        print('epoch: ', i)
        model.train()
        with tqdm(total = num_samples) as pbar:
            time.sleep(0.05)            
            for _i,data in enumerate(data_loader,0):
#                For debugging, please run the line below
#                _i,data = next(iter(enumerate(params.reader.get_train(iterable = True, shuffle = True),0)))

                b_inputs = data[0].to(params.device)
                b_targets = data[-1].to(params.device)
                
                # Does not train if batch_size is 1, because batch normalization will crash
                if b_inputs.shape[0] == 1:
                    continue
                optimizer.zero_grad()
                if len(unitary_params)>0:
                    unitary_optimizer.zero_grad()

                outputs = model(b_inputs)

                outputs = outputs.view(-1, params.output_dim)
                b_targets = b_targets.view(-1, params.output_dim)
                loss = criterion(outputs, b_targets.argmax(dim = -1))
    
                loss.backward()
                optimizer.step()
                if len(unitary_params)>0:
                    unitary_optimizer.step()
                    
                n_total = len(outputs)
                n_correct = (outputs.argmax(dim = -1) == b_targets.argmax(dim = -1)).sum().item()
                train_acc = n_correct/n_total 

                #Update Progress Bar
                pbar.update(params.batch_size)
                ordered_dict={'acc': train_acc, 'loss':loss.item()}        
                pbar.set_postfix(ordered_dict=ordered_dict)

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='running experiments on multimodal datasets.')
    parser.add_argument('-config', action = 'store', dest = 'config_file', help = 'please enter configuration file.',default = 'config/run.ini')
    args = parser.parse_args()
    params = Params()
    params.parse_config(args.config_file) 
    params.config_file = args.config_file
    set_seed(params)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    run(params)
        
        
   