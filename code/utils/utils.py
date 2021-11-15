import json
import copy

from os import listdir
from os.path import isfile, join, isdir
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import numpy as np 
from copy import deepcopy
#from gensim.models import KeyedVectors


PAD_INDEX = 0
UNK_INDEX = 1
EOS_INDEX = 2
GO_SYMBOL_INDEX = 3
EMPTY_INDEX = 4
EOU_INDEX = 5 #end of utterance
CLS_INDEX = 7 #end of utterance
SEPARATOR_INDEX = 6 #end of utterance

special_tokens_ids = list(range(6))

PAD = "PAD"
UNK = "UNK"
EOS = "EOS"
GO_SYMBOL = "GO_SYMBOL"
EMPTY = "EMPTY"
EOU = "EOU"
CLS = "CLS"
SEPARATOR = "<SEP>"
def tokenize(text):
    return word_tokenize(text.lower())

def get_json_files(folder):
#    return [folder+f for f in listdir(folder) if (isfile(join(folder, f)) and f.endswith('.json'))]
    return [folder+f+"/"+f+".json" for f in listdir(folder) if (isdir(join(folder, f)))]

def read_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
        return data

def read_flowchart_jsons(flowchart_dir):
    # load flowcharts from file
    flowcharts = {}
    flowchart_files = get_json_files(flowchart_dir)
    for flowchart_file in flowchart_files:
        flowchart_json = read_json(flowchart_file)
        flowcharts[flowchart_json['name']] = flowchart_json
    return flowcharts

def read_flowchart_doc_jsons(flowchart_dir):
    # load flowcharts from file
    flowchart_docs = {}
    flowchart_doc_files = [(f,flowchart_dir+f+"/supporting_docs.json") for f in listdir(flowchart_dir) if (isdir(join(flowchart_dir, f)))]
    for f,doc in flowchart_doc_files:
        doc_json = read_json(doc)
        flowchart_docs[f] = doc_json
    return flowchart_docs

def read_dialog_jsons(dialog_dir):
    # load train, dev and test dialogs from file
    trnJson = read_json(dialog_dir + "trn.json")
    valJson = read_json(dialog_dir + "val.json")
    tstJson = read_json(dialog_dir + "tst.json")
    return trnJson, valJson, tstJson

def get_vocab_from_dialog_json(dialog_json, only_agent=False):
    vocab = set([])
    max_length = 0
    for dialog in dialog_json['dialogs']:
        for utt in dialog['utterences']:
            if only_agent and utt['speaker']=="user":
                continue
            tokens = tokenize(utt['utterance'])
            max_length = len(tokens) if len(tokens) > max_length else max_length
            vocab = vocab.union(tokens)
    return vocab, max_length

def vectorize_text(text, vocab, length):
    vectorized_text = [PAD_INDEX]*length
    tokens = tokenize(text)
    if len(tokens) == 0:
        tokens = [EMPTY]
    for idx, word in enumerate(tokens[:length-1]):
        vectorized_text[idx] = vocab[word] if word in vocab else UNK_INDEX
    vectorized_text[min(len(tokens), length-1)] = EOS_INDEX
    return min(len(tokens)+1,length), vectorized_text

def build_vocab(flowcharts, trnJson, valJson):
    
    glob = {}

    vocab_to_idx = {}
    vocab_to_idx[PAD] = PAD_INDEX
    vocab_to_idx[UNK] = UNK_INDEX
    vocab_to_idx[EOS] = EOS_INDEX
    vocab_to_idx[GO_SYMBOL] = GO_SYMBOL_INDEX
    vocab_to_idx[EMPTY] = EMPTY_INDEX
    vocab_to_idx[EOU] = EOU_INDEX
    vocab_to_idx[CLS] = CLS_INDEX
    vocab_to_idx[SEPARATOR] = SEPARATOR_INDEX
    n_special_tokens = len(vocab_to_idx)

    encoder_vocab = set([])
    max_edge_label_length = 0
    max_node_utterance_length = 0

    for _, flowchart in flowcharts.items():
        for _, node_properties in flowchart['nodes'].items():
            tokens = tokenize(node_properties['utterance'])
            encoder_vocab = encoder_vocab.union(tokens)
            max_node_utterance_length = len(tokens) if len(tokens) > max_node_utterance_length else max_node_utterance_length
        for _, edges in flowchart['edges'].items():
            for edge_text, _ in  edges.items():
                tokens = tokenize(edge_text)
                encoder_vocab = encoder_vocab.union(tokens)
                max_edge_label_length = len(tokens) if len(tokens) > max_edge_label_length else max_edge_label_length
    
    glob['max_node_utterance_length'] = max_node_utterance_length + 1 # +1 is for the EOS token
    glob['max_edge_label_length'] = max_edge_label_length + 1 # +1 is for the EOS token
    glob['max_flowchart_text_length'] = max(max_edge_label_length + 1, max_node_utterance_length + 1) # +1 is for the EOS token

    max_input_sent_length = 0
    vocab, max_length = get_vocab_from_dialog_json(trnJson)
    encoder_vocab = encoder_vocab.union(vocab)
    max_input_sent_length = max_length if max_length > max_input_sent_length else max_input_sent_length

    vocab, max_length = get_vocab_from_dialog_json(valJson)
    encoder_vocab = encoder_vocab.union(vocab)
    max_input_sent_length = max_length if max_length > max_input_sent_length else max_input_sent_length

    encoder_vocab_to_idx = copy.deepcopy(vocab_to_idx)
    for idx, word in enumerate(encoder_vocab):
        encoder_vocab_to_idx[word] = idx+n_special_tokens
    encoder_idx_to_vocab = {v: k for k, v in encoder_vocab_to_idx.items()}

    glob['encoder_vocab_to_idx'] = encoder_vocab_to_idx
    glob['encoder_idx_to_vocab'] = encoder_idx_to_vocab
    glob['max_input_sent_length'] = max_input_sent_length+1 # +1 is for the EOS token

    decoder_vocab = set([])
    max_response_sent_length = 0

    vocab, max_length = get_vocab_from_dialog_json(trnJson, only_agent=True)
    decoder_vocab = decoder_vocab.union(vocab)
    max_response_sent_length = max_length if max_length > max_response_sent_length else max_response_sent_length

    vocab, max_length = get_vocab_from_dialog_json(valJson, only_agent=True)
    decoder_vocab = decoder_vocab.union(vocab)
    max_response_sent_length = max_length if max_length > max_response_sent_length else max_response_sent_length

    decoder_vocab_to_idx = copy.deepcopy(vocab_to_idx)
    for idx, word in enumerate(decoder_vocab):
        decoder_vocab_to_idx[word] = idx+n_special_tokens
    decoder_idx_to_vocab = {v: k for k, v in decoder_vocab_to_idx.items()}

    glob['decoder_vocab_to_idx'] = decoder_vocab_to_idx
    glob['decoder_idx_to_vocab'] = decoder_idx_to_vocab
    glob['max_response_sent_length'] = max_response_sent_length + 1 # +1 is for the EOS token

    return glob

def create_batches(data, batch_size):
    
    size = data.size
    batches = zip(range(0, size - batch_size, batch_size),
                  range(batch_size, size, batch_size))
    batches = [(start, end) for start, end in batches]
    # last batch
    if batches[-1][1] < size:
        batches.append((batches[-1][1], size))
    return batches
'''
def get_embedding_matrix_from_pretrained(glob, args, dictionary = 'encoder_vocab_to_idx'):
    words = list(glob[dictionary].keys())
    model = KeyedVectors.load_word2vec_format(args.w2v_path, binary=True)
    matrix_len = len(words)
    weights_matrix = np.zeros((matrix_len, model.vector_size))
    words_found = 0

    for i, word in enumerate(words):
        try: 
            weights_matrix[i,:] = model[word]
            words_found += 1
        except KeyError:
            weights_matrix[i,:] = np.random.normal(scale=0.6, size=(model.vector_size, ))
    return weights_matrix
'''
def get_embedding_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.tensor(deepcopy(weights_matrix))})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer

def get_indexes_for_bleu(labels):
    start_idxs = np.where(np.array(labels)==1)[0]
    end_idxs = start_idxs[1:]-1
    end_idxs=np.append(end_idxs,len(labels)-1)
    idxs = [(start_idxs[i],end_idxs[i]+1) for i in range(len(start_idxs))]#+1 for end idx because [start:end+1] will select elements from start and including end
    return idxs
