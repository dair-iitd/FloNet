# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket
import itertools
import numpy as np

import torch
import random
from transformers import cached_path
import torch.nn.functional as F
import warnings

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"

logger = logging.getLogger(__file__)

def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(archive, tempdir)
    return tempdir


def get_dataset(tokenizer, dataset_path, dataset_cache):
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # To avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir

# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
GENERATION_INPUTS = ["valid_response", "generation_persona", "generation_history"]
logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def pad_generation_dataset(dataset,padding=0,use_flowchart=True):
    oneD_lists = ["valid_response","generation_persona_lengths","generation_history_lengths"]
    for name in oneD_lists:
        max_l = max(len(x) for x in dataset[name])
        dataset[name] = [x + [padding] * (max_l - len(x)) for x in dataset[name]]
    
    twoD_lists = [ "generation_persona", "generation_history"]
    if not use_flowchart:
        twoD_lists.remove("generation_persona")

    for name in twoD_lists:
        max_list_len = max([len(x) for x in dataset[name]])
        max_utt_len = 0
        for item in dataset[name]:
            max_utt_len=max(max_utt_len,max([len(x) for x in item]))

        for i,item in enumerate(dataset[name]):
            dataset[name][i]=[x + [padding] * (max_utt_len - len(x)) for x in item]

        dataset[name]=[x + [[padding]*max_utt_len]*(max_list_len-len(x)) for x in dataset[name]]
    return dataset

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def edit_segment_lengths(sequence, p_len, h_len):
    sequence[0]=sequence[0][:p_len]
    a = -1
    for a in range(1,len(sequence[1:-1])+1,2):
        if len(list(chain(*sequence[a:-1])))<=h_len:
            break
    
    if a ==-1:
        return sequence
        
    if a==len(sequence)-2 and len(list(sequence[a]))>h_len:
        sequence = [sequence[0]] + [[sequence[a][0]]+sequence[a][-h_len+1:]] + [sequence[-1]]
    else:
        sequence = [sequence[0]] + sequence[a:-1] + [sequence[-1]]
    return sequence

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True,max_input_length=512, personality_length = 200, history_length = 300):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

    #trim to max_input_length
    sequence = edit_segment_lengths(sequence, personality_length, history_length)

    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

def get_data_loaders(args, tokenizer,use_flowchart=True,return_dataset=False):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    #logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list), "test": defaultdict(list)}    
    generation_datasets = {"valid": defaultdict(list), "test": defaultdict(list)}    
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy() if use_flowchart else []
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels, max_input_length=args.max_input_length,personality_length = args.personality_length, history_length = args.history_length)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates

                    if dataset_name != "train":
                        generation_datasets[dataset_name]["valid_response"].append(utterance["candidates"][-1]) #last candidate is the correct one
                        generation_datasets[dataset_name]["generation_persona"].append(persona) #only needed for generation
                        generation_datasets[dataset_name]["generation_history"].append(history) #only needed for generation
                        generation_datasets[dataset_name]["generation_n_persona"].append(len(persona)) #only needed for generation
                        generation_datasets[dataset_name]["generation_n_history"].append(len(history)) #only needed for generation
                        generation_datasets[dataset_name]["generation_persona_lengths"].append( [len(x) for x in persona] if use_flowchart else []) #only needed for generation
                        generation_datasets[dataset_name]["generation_history_lengths"].append([len(x) for x in history]) #only needed for generation
                if use_flowchart:
                    persona = [persona[-1]] + persona[:-1]  # permuted personalities

    #logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": [], "test": []}
    tensor_generation_datasets = {"valid": [], "test": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    for dataset_name, dataset in generation_datasets.items():
        dataset = pad_generation_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]), use_flowchart=use_flowchart)
        for input_name in dataset.keys():
            tensor = torch.tensor(dataset[input_name])
            tensor_generation_datasets[dataset_name].append(tensor)

    logger.info("Build train, validation and test dataloaders")
    train_dataset, valid_dataset, test_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"]), TensorDataset(*tensor_datasets["test"])
    valid_gen_dataset, test_gen_dataset = TensorDataset(*tensor_generation_datasets["valid"]), TensorDataset(*tensor_generation_datasets["test"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size, shuffle=False)
    
    valid_gen_sampler = torch.utils.data.distributed.DistributedSampler(valid_gen_dataset) if args.distributed else None
    test_gen_sampler = torch.utils.data.distributed.DistributedSampler(test_gen_dataset) if args.distributed else None
    valid_gen_loader = DataLoader(valid_gen_dataset, sampler=valid_gen_sampler, batch_size=args.valid_batch_size, shuffle=False)
    test_gen_loader = DataLoader(test_gen_dataset, sampler=test_gen_sampler, batch_size=args.test_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    logger.info("Test dataset (Batch, Candidates, Seq length): {}".format(test_dataset.tensors[0].shape))

    if return_dataset:
        return train_dataset, valid_dataset, test_dataset

    return train_loader, valid_loader, test_loader, train_sampler, valid_sampler, test_sampler, valid_gen_loader, test_gen_loader, valid_gen_sampler, test_gen_sampler

def convert_to_gpt_input(dataset, args, tokenizer, use_flowchart=True):
    """ Prepare the dataset for training and evaluation """
    #logger.info("Build inputs and labels from siamese entry")
    gpt_dataset = defaultdict(list)
    num_candidates = len(dataset[0]["utterances"][0]["candidates"])
    #if args.num_candidates > 0 and dataset_name == 'train':
    num_candidates = min(args.num_candidates, num_candidates)
    for dialog in dataset:
        persona = dialog["personality"].copy() if use_flowchart else []
        for _ in range(args.personality_permutations):
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates-1)
                    instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels, max_input_length=args.max_input_length,personality_length = args.personality_length, history_length = args.history_length)
                    for input_name, input_array in instance.items():
                        gpt_dataset[input_name].append(input_array)
                gpt_dataset["mc_labels"].append(num_candidates - 1)
                gpt_dataset["n_candidates"] = num_candidates

            if use_flowchart:
                persona = [persona[-1]] + persona[:-1]  # permuted personalities

    #logger.info("Pad inputs and convert to Tensor")
    tensor_dataset = []
    gpt_dataset = pad_dataset(gpt_dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
    for input_name in MODEL_INPUTS:
        tensor = torch.tensor(gpt_dataset[input_name])
        if input_name != "mc_labels":
            tensor = tensor.view((-1, gpt_dataset["n_candidates"]) + tensor.shape[1:])
        tensor_dataset.append(tensor)

    return tensor_dataset


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False,personality_length=args.personality_length,history_length=args.history_length)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            n_iter = 0
            while prev.item() in special_tokens_ids and n_iter < args.sample_turns:
                n_iter+=1
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def beam_search(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = [[]]
        current_probs = []
        final_outputs = []
        final_probs = []

    for i in range(args.max_length):
        if i>0 and current_output==[]:
            return final_outputs, final_probs

        if len(final_outputs)>=args.beam:
            continue
        instances = [build_input_from_segments(personality, history, c_out, tokenizer, with_eos=False,personality_length=args.personality_length,history_length=args.history_length) for c_out in current_output]

        input_ids = [x["input_ids"] for x in instances]#all are of same length
        token_type_ids = [x["token_type_ids"] for x in instances]
        input_ids = torch.tensor(input_ids, device=args.device)
        token_type_ids = torch.tensor(token_type_ids, device=args.device)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[:, -1, :] / args.temperature #logits is (beam_size, seq len, probs) so we pick last seq index
        #logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        top_ = torch.topk(probs, args.beam,dim=-1)
        top_probs = top_[0].cpu().numpy()
        top_ids = top_[1].cpu().numpy()
        #remove beams with special tokens if length less than min length
        special_token_positions = [[k for k,y in enumerate(x) if y in special_tokens_ids] for x in top_ids]
        finished_seq_idx = [i for i,x in enumerate(special_token_positions) if len(x)>0]
        if len(finished_seq_idx)>0:
            if i < args.min_length:
                top_ids = [[y for y in x if y not in special_tokens_ids] for x in top_ids]
                top_probs = [[y for k,y in enumerate(x) if k not in special_token_positions[j]] for j,x in enumerate(top_probs)]
            else:
                #get sequences that have finished with the special token
                final_outputs += list(np.array(current_output)[finished_seq_idx])
                final_outputs = [list(x) for x in final_outputs]
                final_probs += list(np.array(current_probs)[finished_seq_idx])

        if i!=0:
            top_probs = [[current_probs[j]*y for y in x] for j,x in enumerate(top_probs)]
            top_ids = [[current_output[j] + [y] for y in x] for j,x in enumerate(top_ids)]
            #flatten
            top_probs = list(itertools.chain.from_iterable(top_probs))
            top_ids = list(itertools.chain.from_iterable(top_ids))
            #get topk
            top_k_idx = np.argsort(top_probs)[::-1][:args.beam]
            top_probs = list(np.array(top_probs)[top_k_idx])
            top_ids = list(np.array(top_ids)[top_k_idx])
            top_ids = [list(x) for x in top_ids]
        if i==0:
            top_ids = [[x] for x in top_ids[0]]
            top_probs = top_probs[0]

        current_output = top_ids
        current_probs = top_probs

    #add outputs to final outputs
    final_outputs +=  current_output 
    final_probs += current_probs
    #pick topk
    top_k_idx = np.argsort(final_probs)[::-1][:args.beam]
    final_probs = list(np.array(final_probs)[top_k_idx])
    final_outputs = list(np.array(final_outputs)[top_k_idx])
    final_outputs = [list(x) for x in final_outputs]
    return final_outputs, final_probs


def convert_to_gpt_input_for_generation(dataset, args, tokenizer, use_flowchart=True):
    """ Prepare the dataset for training and evaluation """
    #logger.info("Build inputs and labels from siamese entry")
    gpt_dataset = defaultdict(list)
    num_candidates = len(dataset[0]["utterances"][0]["candidates"])
    #if args.num_candidates > 0 and dataset_name == 'train':
    num_candidates = min(args.num_candidates, num_candidates)
    for dialog in dataset:
        persona = dialog["personality"].copy() if use_flowchart else []
        for _ in range(args.personality_permutations):
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]
                gpt_dataset["valid_response"].append(utterance["candidates"][-1]) #last candidate is the correct one
                gpt_dataset["generation_persona"].append(persona) #only needed for generation
                gpt_dataset["generation_history"].append(history) #only needed for generation
                gpt_dataset["generation_n_persona"].append(len(persona)) #only needed for generation
                gpt_dataset["generation_n_history"].append(len(history)) #only needed for generation
                gpt_dataset["generation_persona_lengths"].append( [len(x) for x in persona] if use_flowchart else []) #only needed for generation
                gpt_dataset["generation_history_lengths"].append([len(x) for x in history]) #only needed for generation
                

    #logger.info("Pad inputs and convert to Tensor")
    tensor_dataset = []
    gpt_dataset = pad_generation_dataset(gpt_dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]), use_flowchart=use_flowchart)
    for input_name in gpt_dataset.keys():
        tensor = torch.tensor(gpt_dataset[input_name])
        tensor_dataset.append(tensor)

    return tensor_dataset

def process_string(string):
    string=string.strip()
    string=string.replace(".", " .")
    string=string.replace("?", " ?")
    return string

def choose_negative_candidate(history,worst_response):
    #pick agent utterances
    agent_utts = history[1::2]
    negative_candidate = ""
    if random.uniform(0, 1)>0.8 or len(agent_utts)==0:
        negative_candidate = worst_response
    else:
        negative_candidate = random.choice(agent_utts)
    return negative_candidate

def tokenize_for_gpt(entries, tokenizer):
    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    dataset = tokenize(entries)
    return dataset

def pad_inference_input(data,padding=0):
    max_l = max(len(x) for x in data)
    data = [x + [padding] * (max_l - len(x)) for x in data]
    return data