# -*- coding: utf-8 -*-
import torch
import time
from torch import nn
import torch.nn.functional as F
from layers.embedding import PositionEmbedding

from layers.multiply import ComplexMultiply
from layers.mixture import QMixture
from layers.measurement import QMeasurement
from layers.outer import QOuter
from layers.attention import QAttention
from layers.l2_norm import L2Norm

class QAttN(nn.Module):
    def __init__(self, opt):
        super(QAttN, self).__init__()
        self.device = opt.device    
        self.input_dim = opt.input_dim
        self.embed_dim = opt.embed_dim
        self.n_classes = opt.output_dim
        self.amp_projection = nn.Linear(self.input_dim, self.embed_dim) 
        self.phase_embed = PositionEmbedding(self.embed_dim, device = self.device)
        self.multiply = ComplexMultiply()
        self.outer = QOuter()
        self.output_cell_dim = opt.output_cell_dim
        self.out_dropout_rate = opt.out_dropout_rate

        self.attention = QAttention()

        
        #self.dense = QDense(self.embed_dim, self.n_classes)
        self.measurement = QMeasurement(self.embed_dim)

        self.fc_out = nn.Sequential(nn.Linear(self.embed_dim, self.output_cell_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.out_dropout_rate),
                                        nn.Linear(self.output_cell_dim, self.n_classes))
                
    def get_params(self):
        unitary_params = []
        remaining_params = []
        
        remaining_params.extend(list(self.amp_projection.parameters()))
        remaining_params.extend(list(self.phase_embed.parameters()))
            
        unitary_params.extend(list(self.measurement.parameters()))
        remaining_params.extend(list(self.fc_out.parameters()))
            
        return unitary_params, remaining_params
    
    
    def forward(self, x):
        # (batch_size, time_stamps, input_dim)
        batch_size = x.shape[0]
        time_stamps = x.shape[1]
        
        
        # Amplitude Projection
        # May be replaced by amplitude embedding in real tasks
        amp_rep = F.normalize(self.amp_projection(x), dim = -1)
        
        # Phase Projection 
        # May be replaced by phase embedding in real tasks
        phase_rep = self.phase_embed(torch.zeros_like(x[:,:,0],dtype=torch.int64))
        pure_states = self.multiply([amp_rep, phase_rep])

        pure_matrices = self.outer(pure_states)

        #print(len(pure_states))
        #print(pure_states[0].shape)
        #key_states = [torch.stack(s, dim=1) for s in zip(*pure_states)]
       

        in_states = self.attention(pure_matrices, pure_states, torch.ones(batch_size, time_stamps,1))

        output = []
        
        for _h in in_states:
            measurement_probs = self.measurement(_h)
            _output = self.fc_out(measurement_probs)
            output.append(_output)
            
            
        output = torch.stack(output, dim=-2)
        log_prob = F.log_softmax(output, dim=2) # batch, seq_len,  n_classes

        return log_prob

     