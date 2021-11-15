import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils

    
class D_Encoder(nn.Module):
    
    
    def __init__(self, args):
        super(D_Encoder, self).__init__()
        self.context_size = (args.encoder_hidden_size*2) if args.encoder_bidirectional_flag else args.encoder_hidden_size
        self.dense1 = nn.Linear(self.context_size, self.context_size)
        self.num_layers = args.encoder_num_layers
        self.hidden_size = self.context_size
        self.lstm = nn.GRU(self.context_size, args.encoder_hidden_size, batch_first=True, bidirectional=args.encoder_bidirectional_flag, num_layers=self.num_layers)
        
    def forward(self, uembeds, clens):
        uembeds_ = F.relu(self.dense1(uembeds))
        uembeds_packed = rnn_utils.pack_padded_sequence(uembeds_, clens, batch_first=True, enforce_sorted=False)
        output,_ = self.lstm(uembeds_packed)
        output, r_ulens = rnn_utils.pad_packed_sequence(output, batch_first=True,total_length=uembeds.shape[-2])
        output = output[list(range(output.shape[0])),clens-1,:]
        return output

