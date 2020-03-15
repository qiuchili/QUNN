# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from layers.embedding import PhaseEmbedding
from layers.multiply import ComplexMultiply
from layers.rnn import QRNNCell
from layers.measurement import QMeasurement
from layers.outer import QOuter
from layers.simplenet import SimpleNet
from layers.l2_norm import L2Norm
from layers.dense import QDense
from layers.dropout import QDropout
from layers.activation import QActivation

class QNN(nn.Module):
    def __init__(self, opt):
        super(QNN, self).__init__()
        self.device = opt.device    
        self.input_dim = opt.input_dim
        self.embed_dim = opt.embed_dim
        self.n_classes = opt.output_dim
        self.amp_projection = nn.Linear(self.input_dim, self.embed_dim) 
        self.phase_projection = nn.Linear(self.input_dim, self.embed_dim)

        self.multiply = ComplexMultiply()
        self.outer = QOuter()
        self.out_dropout_rate = opt.out_dropout_rate
        self.num_layers = opt.num_layers
        self.recurrent_cells = nn.ModuleList([QRNNCell(self.embed_dim)]*self.num_layers)
        self.out_dropout = QDropout(p=self.out_dropout_rate)
        self.activation = QActivation(scale_factor = 2, beta = 0.8)
        
        self.dense = QDense(self.embed_dim, self.n_classes)
        self.measurement = QMeasurement(self.n_classes)

        
    def get_params(self):
    	# Get the unitary and non-unitary parameters for the model
        unitary_params = []
        
        for i in range(self.num_layers):
            unitary_params.append(self.recurrent_cells[i].unitary_x)
            unitary_params.append(self.recurrent_cells[i].unitary_h)
            
        #unitary_params.append(self.dense.weight)
        unitary_params.append(self.dense.weight)

        unitary_params.extend(list(self.measurement.parameters()))
     
        remaining_params = list(self.amp_projection.parameters())
        remaining_params.extend(list(self.phase_projection.parameters()))
        for i in range(self.num_layers):
            remaining_params.append(self.recurrent_cells[i].Lambda)
            
       	remaining_params.append(self.dense.Lambda)
            
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
        phase_rep = self.phase_projection(x)
        pure_states = self.multiply([amp_rep, phase_rep])
        pure_matrices = self.outer(pure_states)
        
        in_states = pure_matrices
        for l in range(self.num_layers):
            # Initialize the cell h
            h_r = torch.stack(batch_size*[torch.eye(self.embed_dim)/self.embed_dim],dim =0)
            h_i = torch.zeros_like(h_r)
            h = [h_r.to(self.device),h_i.to(self.device)]
            all_h = []
            for t in range(time_stamps):
                h = self.recurrent_cells[l](in_states[t],h)
                h = self.activation(h)
                all_h.append(h)
            in_states = all_h

        output = []
        
        for _h in in_states:
            _h = self.out_dropout(_h)
            _h = self.dense(_h)
            measurement_probs = self.measurement(_h)
            output.append(measurement_probs)
            
            
        output = torch.stack(output, dim=-2)
        log_prob = torch.log(output) # batch, seq_len,  n_classes

        return log_prob