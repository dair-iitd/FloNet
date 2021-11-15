import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils
import pickle
from utils import get_embedding_layer#, get_embedding_matrix_from_pretrained
    
class U_Encoder(nn.Module):
    
    
    def __init__(self,args, glob):
        super(U_Encoder, self).__init__()
        self.embedding_size = args.emb_size
        self.encoder_vocab_size = args.encoder_vocab_size
        self.num_layers = args.encoder_num_layers
        self.hidden_size = args.encoder_hidden_size
        #self.emb_lookup = nn.Embedding(self.encoder_vocab_size, self.embedding_size)
        self.emb_lookup = get_embedding_layer(self.load_embedding_matrix(args),non_trainable=False)
        self.lstm = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=args.encoder_bidirectional_flag, num_layers=self.num_layers,dropout=args.dropout)
        
    def forward(self, utterances, ulens):
        uembeds = self.emb_lookup(utterances)
        uembeds_packed = rnn_utils.pack_padded_sequence(uembeds, ulens, batch_first=True, enforce_sorted=False)
        output,_ = self.lstm(uembeds_packed) #batch,seq_len,num_dir*hidden
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True,total_length=utterances.shape[-1])
        output = output[list(range(output.shape[0])),ulens-1,:]#batch,num_dir*hidden
        return output

    def load_embedding_matrix(self,args):
        with open(args.saved_glove_path,"rb") as f:
            matrix = pickle.load(f)
            num_embeddings, embedding_dim = matrix.shape
            assert embedding_dim == args.emb_size
            assert num_embeddings == args.encoder_vocab_size
            return matrix