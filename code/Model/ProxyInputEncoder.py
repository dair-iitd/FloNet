import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils

from .Decoder import Decoder
from .D_Encoder import D_Encoder
from .U_Encoder import U_Encoder

class ProxyInputEncoder(nn.Module):
    
    def __init__(self, args, glob):
        super(ProxyInputEncoder, self).__init__()
        self.cuda = args.cuda
        #context encoder
        self.utterance_encoder=U_Encoder(args, glob)
        self.context_encoder=D_Encoder(args)
    
    def _encode_utterances(self, contexts, context_utterance_lengths):

        batch_size, num_utts_in_context, num_words = contexts.shape
        #Reshape to 2d tensor
        contexts_2d = contexts.contiguous().view(-1, num_words)#(batch*num_utts_in_context,num_words)
        context_utterance_lengths_2d = context_utterance_lengths.view(-1)

        #Encode the utterances
        mask = context_utterance_lengths_2d>0
        contexts_2d_selected = contexts_2d[mask]
        context_utterance_lengths_2d_selected = context_utterance_lengths_2d[mask]
        c_utterance_embeds_selected = self.utterance_encoder(contexts_2d_selected, context_utterance_lengths_2d_selected)
        if self.cuda == True:
            c_utterance_embeds_2d = torch.zeros(contexts_2d.shape[0], c_utterance_embeds_selected.shape[1]).cuda()
        else:
            c_utterance_embeds_2d = torch.zeros(contexts_2d.shape[0], c_utterance_embeds_selected.shape[1])
        c_utterance_embeds_2d.masked_scatter_(mask.unsqueeze(1), c_utterance_embeds_selected)
        assert c_utterance_embeds_2d.shape[0] == batch_size*num_utts_in_context
        
        #Reshape and encode the contexts to get context embeddings
        c_utterance_embeds = c_utterance_embeds_2d.view(batch_size, num_utts_in_context, -1)
        return c_utterance_embeds

    def get_context_embedding(self, contexts, context_utterance_lengths, context_lengths):
        #get embedding of utterances, c_utterance_embeds=(batch size, num utterances, encoding size)
        c_utterance_embeds=self._encode_utterances(contexts, context_utterance_lengths)

        #handle empty contexts
        context_mask=context_lengths>0
        c_utterance_embeds_masked=c_utterance_embeds[context_mask]
        context_lengths_masked=context_lengths[context_mask]
        context_embeds_masked = self.context_encoder(c_utterance_embeds_masked, context_lengths_masked)
        context_embeds_masked=context_embeds_masked.squeeze(1)
        if self.cuda == True:
            context_embeds = torch.zeros(c_utterance_embeds.shape[0],context_embeds_masked.shape[1]).cuda()
        else:
            context_embeds = torch.zeros(c_utterance_embeds.shape[0],context_embeds_masked.shape[1])
        context_embeds.masked_scatter_(context_mask.unsqueeze(1), context_embeds_masked)
        return context_embeds

    def forward(self,contexts,context_utterance_lengths,context_lengths):
        input_embeddings = self.get_context_embedding( contexts,context_utterance_lengths,context_lengths)
        
        return input_embeddings
