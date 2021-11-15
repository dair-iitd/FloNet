import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils

from .Decoder import Decoder
from .D_Encoder import D_Encoder
from .U_Encoder import U_Encoder

class Input_Encoder(nn.Module):
    
    def __init__(self, args, glob):
        super(Input_Encoder, self).__init__()

        #context encoder
        self.utterance_encoder=U_Encoder(args, glob)
        self.context_encoder=D_Encoder(args)
        self.query_encoder=U_Encoder(args, glob)

        self.cuda=args.cuda
        #self.memn2n = MemoryN2N(args)
        #self.decoder = Decoder(args)
    
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

    def get_query_embedding(self,queries,query_lengths):#
        #Encode the utterances
        mask = query_lengths>0
        queries_selected = queries[mask]
        query_lengths_selected = query_lengths[mask]
        query_embeds_selected = self.query_encoder(queries_selected, query_lengths_selected)
        if self.cuda == True:
            query_embeds = torch.zeros(queries.shape[0], query_embeds_selected.shape[1]).cuda()
        else:
            query_embeds = torch.zeros(queries.shape[0], query_embeds_selected.shape[1])
        query_embeds.masked_scatter_(mask.unsqueeze(1), query_embeds_selected)

        return query_embeds

    def forward(self,contexts,context_utterance_lengths,context_lengths,\
        queries,query_lengths):

        ######
        ## Method 1 of handing queries and contexts:
        ## Pass context+queries as a single seqeuence of utterances passed to HRED's encoder
        s1,s2,s3=contexts.shape
        combined_contexts=torch.zeros(s1,s2+1,s3).to(contexts)
        for i in range(context_lengths.shape[0]):
            idx=context_lengths[i]
            combined_contexts[i] = torch.cat([contexts[i,:idx,:], queries[i].unsqueeze(0), contexts[i,idx:,:]], 0)

        combined_context_utterance_lengths = torch.cat((context_utterance_lengths,query_lengths.unsqueeze(1)),axis=1)
        combined_context_lengths=context_lengths+1
        input_embeddings = self.get_context_embedding( combined_contexts, combined_context_utterance_lengths, combined_context_lengths)
        
        return input_embeddings
