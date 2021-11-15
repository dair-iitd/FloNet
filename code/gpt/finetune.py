# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
import time
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
import numpy as np
from nltk.tokenize import word_tokenize

import torch
from torchtext.data.metrics import bleu_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME, GPT2LMHeadModel, OpenAIGPTLMHeadModel)
from ignite.handlers import Timer
from torch.nn import CrossEntropyLoss, Softmax

from .gpt_utils import get_dataset, make_logdir, download_pretrained_model, add_special_tokens_, get_data_loaders, average_distributed_scalar, pad_dataset, sample_sequence, top_filtering, beam_search

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../../../../transfer_transfero_flowchart/data/flowchart_data_out_domain_correct_utterance_personality_SMALL.json", help="Path or url of the dataset. If empty download from S3.")#TODO
    parser.add_argument("--dataset_cache", type=str, default='../../../../transfer_transfero_flowchart//cache/flowchart_data_out_domain_correct_utterance_personality_SMALL_cache', help="Path or url of the dataset cache")#TODO
    parser.add_argument("--use_flowchart", type=bool, default=True, help="Use flowchart or not")#TODO
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")#TODO
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt#TODO
    parser.add_argument("--suffix", type=str, default="", help="Suffix for folder that saves model checkpoint and logs")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=8, help="Number of previous exchanges to keep in history")#TODO
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=4, help="Number of training epochs")#TODO
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")

    #for getting results
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")

    parser.add_argument("--sample_turns", type=int, default=10, help="number of time to sample for next token (repeated sampling if token is a special token)")#TODO
    parser.add_argument("--max_input_length", type=int, default=510, help="length of acceptable input. 512 for transfertransfero and 1024 for gpt2")#TODO
    parser.add_argument("--personality_length", type=int, default=190, help="length of acceptable flowchart input segment")#TODO
    parser.add_argument("--history_length", type=int, default=200, help="length of acceptable history input segment")#TODO
    parser.add_argument("--save_metric", type=str, default='BLEU', help="nll or BLEU")#TODO
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    return args

def init_gpt_model(args,logger):
    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    if args.gpt_model_checkpoint == "":
        if args.model == 'gpt2':
            args.gpt_model_checkpoint = 'gpt2' #TODO fix this to a pretrained checkpoint
        #raise ValueError("Interacting with GPT2 requires passing a finetuned gpt_model_checkpoint")
        else:
            args.gpt_model_checkpoint = download_pretrained_model()
    #print(args.gpt_model_checkpoint)

    tokenizer_class = GPT2Tokenizer if args.model == 'gpt2' else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained(args.gpt_model_checkpoint)

    model_class = GPT2DoubleHeadsModel if args.model == 'gpt2' else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.gpt_model_checkpoint)
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.gpt_lr, correct_bias=True)
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    return model, optimizer, tokenizer

# Training function and trainer
cross_entropy = CrossEntropyLoss(reduce=False)

def train_gpt(batch,model,gpt_optimizer,step_num,args):
    model.train()
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
    input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, *_ = batch
    (lm_loss), (mc_loss), logits, mc_logits, *_ = model(
        input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
        mc_labels=mc_labels, lm_labels=lm_labels
    )
    #get gpt probs
    probs = get_lm_probs(logits, lm_labels)

    return probs, mc_loss

def get_lm_probs(logits, targets, ignore_index=-100):
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    fct_loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss = -fct_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
    out = loss.view(targets.shape).contiguous()
    out = out[:,-1,:].sum(-1)
    return out

def get_probs(logits, targets, ignore_index=-100):
    bs, n_cat = logits.shape[0], logits.shape[-1]
    softmax = Softmax(-1)
    logits_ = softmax(logits)
    logits_, targets_ = logits_[:,-1,:,:].reshape(-1,n_cat),targets[:,-1,:].reshape(-1)
    
    out = torch.zeros_like(targets_, dtype=torch.float)
    for i in range(len(targets_)):#TODO Optimize this
        if targets_[i]==ignore_index:
            continue
        out[i] = logits_[i][targets_[i]]
    out = out.reshape(bs,-1)
    out = torch.sum(out,-1)
    return out

def get_lm_loss_sum(logits, targets, ignore_index=-100):
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    fct_loss = torch.nn.CrossEntropyLoss(reduction='sum')
    loss = fct_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
    return loss

def gpt_inference(batch,model,args):
    model.eval()
    with torch.no_grad():
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        # if we dont send labels to model, it doesnt return losses
        (lm_loss), (mc_loss), logits, mc_logits, *_ = model(
        input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
        mc_labels=mc_labels, lm_labels=lm_labels
        )
        probs = get_lm_probs(logits, lm_labels)
        lm_loss_sum = get_lm_loss_sum(logits, lm_labels)
        return probs, (lm_loss_sum.item(),int(lm_loss_sum/lm_loss)), lm_loss

# Evaluation function and evaluator (evaluator output is the input of the metrics)
def gpt_infer_sequence(personalities, history, model, tokenizer, args, pred_scores, test=False):
    #return "hello world"
    model.eval()
    with torch.no_grad():
        output_beams = []
        output_probs = []
        references = []
        for i, p in enumerate(personalities):
            output_beams_, output_probs_ = beam_search(p, history, tokenizer, model, args)
            #collect for all, sum probabs for same output and figure out the best
            for j,o in enumerate(output_beams_):
                probab = output_probs_[j] * pred_scores[i]
                o = tokenizer.decode(o,skip_special_tokens=True)
                if o in output_beams:
                    output_probs[output_beams.index(o)]+=probab
                else:
                    output_beams.append(o)
                    output_probs.append(probab)

        #pick the best
        best_idx = np.argsort(output_probs)[::-1][0]
        output_ = output_beams[best_idx]

        if test:
            return output_, output_beams, output_probs
        return output_

def gpt_infer_sequence_new(personalities, history, model, tokenizer, args, pred_scores, test=False):
    #uses huggingface decoding method
    assert len(personalities)==1
    out_ids = sample_sequence(personalities[0], history, tokenizer, model, args)
    output_ = tokenizer.decode(out_ids,skip_special_tokens=True)
    return output_

# Evaluation function and evaluator (evaluator output is the input of the metrics)
def gpt_infer_sequence_beam(personalities, history, model, tokenizer, args, pred_scores, test=False,get_max_persona=False,norm=True,gpt_infer=False):
    #return "hello world"
    model.eval()
    with torch.no_grad():
        output_beams = []
        output_beam_tokens = []
        output_probs = []
        references = []
        max_probab = 0
        max_probab_persona = 0

        for i, p in enumerate(personalities):
            output_beams_, output_probs_ = beam_search(p, history, tokenizer, model, args)
            #collect for all, sum probabs for same output and figure out the best
            for j,o in enumerate(output_beams_):
                probab = output_probs_[j] * pred_scores[i]
                o = tokenizer.decode(o,skip_special_tokens=True)
                if o in output_beams:
                    output_probs[output_beams.index(o)]+=probab
                else:
                    output_beams.append(o)
                    output_beam_tokens.append(output_beams_[j])
                    output_probs.append(probab)
                if get_max_persona and probab>max_probab:
                    max_probab_persona  = i
                    max_probab = probab

        #pick the best
        if len(output_probs) == 0 :
            print("Failed for this history")
            print([tokenizer.decode(x) for x in history])
            return " ", [" "], [[0]], [0]
        if norm and gpt_infer:
            output_probs = [np.exp(np.log(np.array(x))/len(output_beam_tokens[i])) for i,x in enumerate(output_probs)]
        elif norm:
            output_probs = [np.exp(np.log(x.cpu().numpy())/len(output_beam_tokens[i])) for i,x in enumerate(output_probs)]
        best_idx = np.argsort(output_probs)[::-1][0]
        output_ = output_beams[best_idx]
        if get_max_persona:
            if output_probs[best_idx]!=max_probab:
                max_probab_persona=-1#combo
            return output_, output_beams, output_beam_tokens, output_probs, max_probab_persona
        if test:
            return output_, output_beams, output_beam_tokens, output_probs
        return output_
