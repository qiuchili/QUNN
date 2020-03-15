# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import numpy as np

def PhaseEmbedding(input_dim, embedding_dim, freeze = False):
    phase_embedding_matrix = torch.empty(input_dim,embedding_dim)
    nn.init.uniform_(phase_embedding_matrix,0, 2*np.pi)
    embedding_layer = nn.Embedding.from_pretrained(phase_embedding_matrix, freeze=freeze)
    return embedding_layer

def FrequencyEmbedding(input_dim, embedding_dim, freeze = False):
    phase_embedding_matrix = torch.empty(input_dim,embedding_dim)
    nn.init.uniform_(phase_embedding_matrix,0, 10)
    embedding_layer = nn.Embedding.from_pretrained(phase_embedding_matrix, freeze=freeze)
    return embedding_layer