import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils
#from utils import get_embedding_layer, get_embedding_matrix_from_pretrained

class Decoder(nn.Module):

    def __init__(self, args, glob):
        super(Decoder, self).__init__()
        self.word_size = args.word_size
        self.hidden_size = args.hidden_size
        self.word2embed = nn.Embedding(args.vocab_size, args.word_size)#get_embedding_layer(get_embedding_matrix_from_pretrained(glob, args, dictionary = 'decoder_vocab_to_idx'))
        self.lstm = nn.GRU(input_size=self.hidden_size*2 + self.word_size, hidden_size=self.hidden_size,
                           batch_first=True,dropout=args.dropout)
        self.worder = nn.Linear(self.hidden_size, args.vocab_size,bias=False)
        self.cuda = args.cuda
        

    def forward(self, cembeds, r_utts, r_ulens):
        num_utts, num_words = r_utts.shape
        rwembeds = self.word2embed(r_utts)
        # We need to provide the current sentences's embedding or "thought" at every timestep.
        cembeds = cembeds.unsqueeze(1).repeat(1, num_words, 1)  # (batchsize, num_words, hidden_size)
        if( self.cuda == True ):
            start_pad = torch.zeros(num_utts, 1, rwembeds.shape[2]).cuda()
        else:
            start_pad = torch.zeros(num_utts, 1, rwembeds.shape[2])
        d_rwembeds = torch.cat([start_pad, rwembeds[:, :-1, :]], dim=1)
        # Supply predicted_embedding and delayed word embeddings for teacher forcing.
        decoder_input = torch.cat([cembeds, d_rwembeds], dim=2)
        decoder_input = rnn_utils.pack_padded_sequence(decoder_input, r_ulens, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(decoder_input)
        output, r_ulens = rnn_utils.pad_packed_sequence(output, batch_first=True,total_length=num_words)
        worddis = self.worder(output)
        # (d_batch_size, maxlen, vocab_size)
        return worddis

    def sample_step(self, cembeds, r_utts, r_ulens):
        num_utts, num_words = r_utts.shape
        rwembeds = self.word2embed(r_utts)
        # We need to provide the current sentences's embedding or "thought" at every timestep.
        cembeds = cembeds.unsqueeze(1).repeat(1, num_words+1, 1)  # (batchsize, num_words+1(for start pad), hidden_size)
        if( self.cuda == True ):
            start_pad = torch.zeros(num_utts, 1, rwembeds.shape[2]).cuda()
        else:
            start_pad = torch.zeros(num_utts, 1, rwembeds.shape[2])
        d_rwembeds = torch.cat([start_pad, rwembeds], dim=1)
        # Supply predicted_embedding and delayed word embeddings for teacher forcing.
        decoder_input = torch.cat([cembeds, d_rwembeds], dim=2)
        decoder_input = rnn_utils.pack_padded_sequence(decoder_input, r_ulens+1, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(decoder_input)
        output, r_ulens = rnn_utils.pad_packed_sequence(output, batch_first=True,total_length=num_words+1)
        worddis = self.worder(output)
        # (d_batch_size, maxlen, vocab_size)
        return worddis

    def greedy(self, contexts, eot, num_steps=50):
        bsize = contexts.shape[0]
        all_sents = []
        predids_mat = torch.zeros(bsize, num_steps).to(contexts)
        input = torch.zeros(bsize,1,self.hidden_size*2+self.word_size).to(contexts)
        input[:,0,:self.hidden_size*2] = contexts
        output, hidden = self.lstm(input)
        worddis = self.worder(output).squeeze(1).detach()
        wids = worddis.max(1)[1]
        predids_mat[:, 0] = wids
        finished = torch.zeros(bsize, dtype=torch.uint8).cuda()
        for iter in range(1, num_steps):
            input[:,0,self.hidden_size*2:] = self.word2embed(wids)
            output, hidden = self.lstm(input, hidden)
            worddis = self.worder(output).squeeze(1)
            wids = worddis.max(1)[1]
            predids_mat[:, iter] = wids
            finished = finished  + (wids==eot)
            predids_mat[finished, iter] = 0

        return predids_mat           


    def forward_first(self, c_embeds):
        assert c_embeds.dim()==2
        nsize = c_embeds.shape[0]
        input = torch.zeros(nsize,1,self.hidden_size*2+self.word_size).to(c_embeds)
        input[:,0,:self.hidden_size*2] = c_embeds
        output, hidden = self.lstm(input)
        worddis = self.worder(output).squeeze()
        worddis = F.softmax(worddis, dim=0).log()
        return hidden, worddis
    
    def forward_next(self, c_embeds, prev_hiddens, prev_wids):
        assert c_embeds.dim()==2
        nsize = prev_hiddens.shape[1]
        input = torch.zeros(nsize,1,self.hidden_size*2+self.word_size).to(c_embeds)
        input[:,0,:self.hidden_size*2] = c_embeds
        input[:,0,self.hidden_size*2:] = self.word2embed(prev_wids)
        output, new_hiddens = self.lstm(input, prev_hiddens)
        worddis = self.worder(output).squeeze(1)
        worddis = F.softmax(worddis, dim=1).log().detach()
        return new_hiddens, worddis

    
    def step(self, contexts, prev_wids=None, prev_hidden=None):
        bsize = contexts.shape[0]
        input = torch.zeros(bsize,1,self.hidden_size*2+self.word_size).to(contexts)
        input[:,0,:self.hidden_size*2] = contexts
        if prev_wids is not None:
            input[:,0,self.hidden_size*2:] = self.word2embed(prev_wids)
        output, hidden = self.lstm(input, prev_hidden)
        worddis = self.worder(output).squeeze(1).detach()
        return worddis, hidden
