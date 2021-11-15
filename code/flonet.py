import numpy as np
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import sys
import io
import os
import time
import copy
import argparse
from Model.ProxyScore import ProxyScore
import nltk
import pickle
import json
from shutil import copyfile
from utils.Flowcharts import Flowcharts
from utils.ProxyScoreData import ProxyScoreData, ProxyScoreBatch
from utils.proxy_scores import get_scores_dict
from utils import read_flowchart_jsons, read_dialog_jsons, build_vocab, create_batches, PAD, get_indexes_for_bleu, read_flowchart_doc_jsons, cache_embedding_matrix
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, WEIGHTS_NAME, CONFIG_NAME
from gpt.gpt_utils import download_pretrained_model, add_special_tokens_, get_data_loaders, convert_to_gpt_input, process_string, choose_negative_candidate, tokenize_for_gpt, convert_to_gpt_input_for_generation, pad_inference_input
from gpt.finetune import init_gpt_model, train_gpt, SPECIAL_TOKENS, gpt_inference, gpt_infer_sequence, gpt_infer_sequence_new, gpt_infer_sequence_beam
from matplotlib import pyplot as plt
import itertools
from pprint import pformat
import logging
logger = logging.getLogger(__file__)
softmax = nn.Softmax(-1)
eps_ = 1e-20
def parge_args():
    parser = argparse.ArgumentParser(description='Train and evaluate FlowNet')

    parser.add_argument('--flowchart-dir', type=str, default='../data/flowcharts/', help='a directiory that contains all flowcharts')
    #change these depending on the dataset
    parser.add_argument('--cached-dialog-path', type=str, default='../data/saved_data/cached_in_domain_dialogs_score_FAQ.pkl', help='cached dataset path')
    parser.add_argument('--domain', type=str, default='in_domain', help='in_domain, out_domain')
    parser.add_argument('--save-name', type=str, default="Combined_test", help='Name of model to be saved')
    parser.add_argument("--retriever_checkpoint", type=str, default="../data/model/Proxybest_checkpoint_ProxyScoreTop1_in_domain_300_0.001_16.pth.tar", help="Path, url or short name of the model")#only for a starting point #calculated on the fly
    parser.add_argument('--dropout_type', type = int, default = 0 )
    parser.add_argument('--dropout_rate', type = int, default = 0 )
    parser.add_argument('--inference', type=int, default=0, help='0 if training, 1 if test')

    ##more or less constant
    parser.add_argument('--load', type=str, default='', help='load from [rerun, dont load a combined model-use the checkpoints of scorer and GPT (empty string)]')
    #calculated on the fly
    parser.add_argument('--dialog-dir', type=str, default='../data/dialogs/', help='a directiory that contains trn, val and tst dialogs')
    parser.add_argument('--cached-scores-path', type=str, default='../data/saved_data/', help='cached dataset path')
    parser.add_argument('--saved-glove-path', type=str, default='./glove6B/', help='cached embedding matrix with glove embeddings')
    parser.add_argument('--ranks-data-save-path', type=str, default="../data/ranks/", help='folder path of accuracy metrics of the dataset')
    #HYPERPARAMETERS-SIAMESE
    parser.add_argument('--lr', type=float, default=.0001, metavar='LR', help='Learning Rate (default: .00015)')
    parser.add_argument('--margin', type=int, default=2, metavar='H', help='margin for loss training')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--hidden-size', type=int, default=300, metavar='H', help='Size of hidden embeddings')
    parser.add_argument('--emb-size', type=int, default=100, metavar='H', help='Size of hidden embeddings')
    parser.add_argument('--encoder_num_layers', type=int, default=1, metavar='H', help='number of layers in encoder')
    parser.add_argument('--bidirectional-encoder', type=bool, default=True, metavar='H', help='bidirectional encoder')
    parser.add_argument('--scorer-topk', type=int, default=5, help='number of documents to send to GPT from scorer')
    #KINDA FIXED-saving complete model
    parser.add_argument('--model-dir', type=str, default='../data/model/', help='dataset')
    parser.add_argument('--log-dir', type=str, default='../logs/', help='save logs of the runs')
    parser.add_argument('--use-transformer', type=bool, default=False)
    #KINDA FIXED-no need to change
    parser.add_argument('--best', action='store_true', default=False, help='Load the best model so far')
    parser.add_argument('--num-epochs', type=int, default=1000, metavar='E', help='Number of epochs for training the model')
    parser.add_argument('--skip-emb-matrix-refresh', type=int, default=0)

    ##GPT ARGS
    parser.add_argument("--gpt_model_checkpoint", type=str, default="/home/cse/staff/shantan1.cstaff/scratch/transfer_transfero_flowchart/openai-gpt_proxy_data_in_domain_cache_BLEU_448773.782979751/", help="Path, url or short name of the model")#only for a starting point #calculated on the fly
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    #CHANGES IN GPT2
    parser.add_argument("--max_history", type=int, default=8, help="Number of previous exchanges to keep in history")
    parser.add_argument("--max_input_length", type=int, default=510, help="length of acceptable input. 512 for transfertransfero and 1024 for gpt2")
    parser.add_argument("--personality_length", type=int, default=190, help="length of acceptable flowchart input segment")
    parser.add_argument("--history_length", type=int, default=300, help="length of acceptable history input segment")
    parser.add_argument("--gpt_batch_size", type=int, default=5, help="Batch size for training")
    #HYPERPARAMETERS-GPT
    parser.add_argument("--beam", type=int, default=5, help="Beam width")
    parser.add_argument("--chart-personality-dropout", type=float, default=0, help="Beam width")
    #KINDA FIXED
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--gpt-lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    #FOR GENERATION-KINDA FIXED
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=5, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--sample_turns", type=int, default=10, help="number of time to sample for next token (repeated sampling if token is a special token)")#TODO
    parser.add_argument("--save_metric", type=str, default='BLEU', help="nll or BLEU")#TODO
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    #calculate folder name based on hyper params for saving model
    rerun = 'rerun'
    param_str = "_".join([args.save_name,str(args.model),args.domain, rerun, str(args.hidden_size), str(args.lr)])
    args.cuda = torch.cuda.is_available()
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.log_dir + param_str):
        os.mkdir(args.log_dir + param_str)
    if not os.path.exists(args.model_dir + param_str):
        os.mkdir(args.model_dir + param_str)

    #calculate additional arguments
    args.load_path = args.model_dir + param_str + '/Scorer_best_checkpoint.pth.tar'
    args.save_path = args.model_dir + param_str + '/Scorer_checkpoint.pth.tar'
    args.best_path = args.model_dir + param_str + '/Scorer_best_checkpoint.pth.tar'
    args.gpt_load_path = args.model_dir + param_str + '/'
    args.gpt_save_path = args.model_dir + param_str + '/GPT_checkpoint/'
    args.gpt_best_path = args.model_dir + param_str + '/GPT_best_checkpoint/'
    args.output_path = args.log_dir + param_str + '/GPT_output.txt'
    args.metric_path = args.log_dir + param_str + '/Net_metrics.txt'
    args.loss_path = args.log_dir + param_str + '/Net_loss.txt'
    args.val_output_path = args.log_dir + param_str + '/val_output.json'
    args.test_output_path = args.log_dir + param_str + '/test_output.json'

    args.dialog_dataset_dir = args.dialog_dir + "dataset/" + args.domain + "/"
    args.dialog_dir = args.dialog_dir + args.domain + "/"
    args.cached_scores_path = args.cached_scores_path + "scored_paths_" + args.domain + "_FAQ.json"
    #args.saved_glove_path = args.saved_glove_path + "saved_embedding_matrix" + str(args.emb_size) +"_" + args.domain + "_dialogs_.pkl"
    args.loss_plot_save_path = args.log_dir + param_str + "/lossPlot.png"
    args.rank_plot_save_path = args.log_dir + param_str + "/rankPlot.png"

    if not os.path.exists(args.gpt_save_path):
        os.mkdir(args.gpt_save_path)
    if not os.path.exists(args.gpt_best_path):
        os.mkdir(args.gpt_best_path)

    print(args)
    return args

def plot_and_save_loss(losses,plot_save_path,name='Loss',test_array = []):
    plt.clf()
    losses_ = np.array(losses)
    test_losses_ = np.array(test_array)
    plt.plot(losses_[:,0], losses_[:,1], label='Train')
    plt.plot(losses_[:,0], losses_[:,2], label='Val')
    if test_array != []:
        plt.plot(test_losses_[:,0], test_losses_[:,1],'o', label='Test')
    plt.title(name)
    plt.legend()
    plt.ylabel(name)
    plt.xlabel('Epochs')
    plt.savefig(plot_save_path)

def plot_scores(scores,plot_save_path,data_save_path):
    scorer = list(itertools.chain.from_iterable([x[0] for x in scores]))
    gpt = list(itertools.chain.from_iterable([x[1] for x in scores]))
    plt.clf()
    plt.hist(gpt,bins=50,color='r')
    plt.hist(scorer,bins=50,color='b')
    plt.savefig(plot_save_path)
    np.savetxt(data_save_path, [scorer,gpt], delimiter=",")

def process_glove_matrix(glob,args):
    glove_matrix_path, missed_idx_path = cache_embedding_matrix(glob,args.emb_size,args.saved_glove_path,args.domain+"_dialogs",args.cached_dialog_path,args.skip_emb_matrix_refresh)
    args.saved_glove_path = glove_matrix_path
    with open(missed_idx_path,'rb') as f:
        training_idxs, words = pickle.load(f) 
        mask = np.ones(len(glob['encoder_vocab_to_idx'])).astype(bool)
        mask[training_idxs]=False
    return args, mask
    
def get_dataset_and_batches():
    logger.info("loading dataset")
    if os.path.exists(args.cached_dialog_path):
        with open(args.cached_dialog_path,"rb") as f:
            trnData, valData, _, glob = pickle.load(f)
    else:
        flowchartsJson = read_flowchart_jsons(args.flowchart_dir)
        flowchartDocsJson = read_flowchart_doc_jsons(args.flowchart_dir)
        trnJson, valJson, tstJson = read_dialog_jsons(args.dialog_dir)
        glob = build_vocab(flowchartsJson, trnJson, valJson)
        flowcharts = Flowcharts(flowchartsJson, glob)
        scores_dict = get_scores_dict(args.dialog_dataset_dir,args.cached_scores_path,args.flowchart_dir)
        trnData = ProxyScoreData(trnJson, flowcharts, glob, scores_dict['train'],flowchartDocsJson)
        valData = ProxyScoreData(valJson, flowcharts, glob, scores_dict['valid'],flowchartDocsJson)
        with open(args.cached_dialog_path,"wb") as f:
            pickle.dump([trnData,valData,glob],f)

    logger.info("batching dataset")
    trn_batches = get_indexes_for_bleu(trnData.batche_start)
    val_batches = get_indexes_for_bleu(valData.batche_start)
    return glob, trnData, valData, trn_batches, val_batches

def set_additional_args(args):
    #for siamese
    args.encoder_vocab_size=len(glob['encoder_vocab_to_idx'])
    args.encoder_hidden_size=args.hidden_size
    args.encoder_bidirectional_flag=args.bidirectional_encoder
    return args

def get_bleu(bleu_data):
    hypothesis = [x[0] for x in bleu_data]
    references = [x[1] for x in bleu_data]
    hypothesis=[nltk.word_tokenize(x) for x in hypothesis]
    references = [[nltk.word_tokenize(x)] for x in references]
    bleu=nltk.translate.bleu_score.corpus_bleu(references, hypothesis)
    #remove padding 
    return bleu

def save_tokenizer(args,gpt,gpt_tokenizer,path):
    torch.save(args, path + '/model_training_args.bin')
    getattr(gpt, 'module', gpt).config.to_json_file(os.path.join(path, CONFIG_NAME))
    gpt_tokenizer.save_pretrained(path)

def init_model(args):
    logger.info("prepare scorer")
    scorer = ProxyScore(args, glob)
    if args.cuda:
        scorer.cuda()
    scorer_optimizer = torch.optim.Adam(params=scorer.parameters(), lr=args.lr)
    #load model
    if args.load!='':
        args.retriever_checkpoint = args.load_path
        args.gpt_model_checkpoint = args.gpt_best_path if os.path.exists(os.path.join(args.gpt_best_path,WEIGHTS_NAME)) else args.gpt_save_path

    if args.retriever_checkpoint!="":
        print("Loading siamese checkpoint from:", args.retriever_checkpoint)
        checkpoint = torch.load(args.retriever_checkpoint, map_location=lambda storage, loc: storage)
        scorer.load_state_dict(checkpoint['state_dict'])
        scorer_optimizer.load_state_dict(checkpoint['optimizer'])
    
    gpt, gpt_optimizer, gpt_tokenizer = init_gpt_model(args,logger)

    ##save tokenizer
    save_tokenizer(args,gpt,gpt_tokenizer,args.gpt_best_path)
    save_tokenizer(args,gpt,gpt_tokenizer,args.gpt_save_path)

    return scorer, scorer_optimizer, gpt, gpt_optimizer, gpt_tokenizer

def make_gpt_input_json(history, response, chart_responses, worst_response):
    entries = []
    for i,r in enumerate(chart_responses):
        negative_candidate = process_string(choose_negative_candidate(history, worst_response))
        entry = {"personality": [process_string(r)],
                "utterances":[{"candidates":[negative_candidate,response],
                    "history":history}]
                }
        entries.append(entry)
    return entries

def get_ranking_info(gt_labels, pred_scores):
    #ranking info
    gt_indexes = np.where(np.array(gt_labels)==1)[0]
    ranked_idxs = np.argsort(pred_scores.detach().cpu().numpy())
    gt_index = [x for x in ranked_idxs if x in gt_indexes][0]#get the first match when multiple GT (happens when multiple nodes have same utterances)
    #ordered_responses = np.array(chart_responses)[ranked_idxs]
    ranking_info = [ranked_idxs, gt_index]
    correct_r1 = gt_index==ranked_idxs[0]
    correct_r5 = gt_index in ranked_idxs[:5]
    return ranking_info, correct_r1, correct_r5

def get_p_dropouts(pred_scores):
    p = pred_scores.detach().cpu().numpy()
    dropout = np.zeros(p.shape[0])
    if args.dropout_type==1:
        dr = args.dropout_rate/100
        dropout = np.random.choice([1,0],pred_scores.shape[0],p=[dr,1-dr])
    elif args.dropout_type==2:
        for i in range(p.shape[0]):
            dr = p[i]*args.dropout_rate/100
            dropout[i]=np.random.choice([1,0],p=[dr,1-dr])
    return dropout

def dropout_entries(entries,pred_scores):
    dropouts = get_p_dropouts(pred_scores).astype(bool)
    #print(dropouts)
    for i, drop in enumerate(dropouts):
        if drop:
            entries[i]['utterances'][0]['history']=[]
    return entries

def train(startIdx, endIdx, mask):
    
    scorer.train()
    batch_entry = ProxyScoreBatch(trnData, glob, startIdx, endIdx)
    
    contexts,  context_utterance_lengths, context_lengths  = torch.as_tensor(batch_entry.context).long(),  torch.as_tensor(batch_entry.context_utterance_lengths).long(),  torch.as_tensor(batch_entry.context_num_utterances).long()
    paths,  path_utterance_lengths, path_lengths  = torch.as_tensor(batch_entry.path).long(),  torch.as_tensor(batch_entry.path_utterance_lengths).long(),  torch.as_tensor(batch_entry.path_num_utterances).long()
    scores = torch.as_tensor(batch_entry.score,dtype=float)
    responses, chart_responses = batch_entry.response_as_text, batch_entry.chart_response_as_text
    gt_labels = batch_entry.gt_label
    assert len(list(set(responses)))==1 #make sure the batch belongs to the same context-response pair

    if args.cuda:
        contexts,  context_utterance_lengths, context_lengths  = contexts.cuda(),  context_utterance_lengths.cuda(), context_lengths.cuda()
        paths, path_utterance_lengths, path_lengths  = paths.cuda(),  path_utterance_lengths.cuda(), path_lengths.cuda()
        scores = scores.cuda()
        
    pred_scores_ = scorer.get_scores(contexts,context_utterance_lengths,context_lengths,paths, path_utterance_lengths, path_lengths)
    #convert to a format such that greater score=better
    topk_idxs = np.argsort(pred_scores_.detach().cpu().numpy())[:args.scorer_topk]#fetch top k
    pred_scores = pred_scores_[list(topk_idxs)]
    pred_scores = softmax(-pred_scores)    

    ####GPT####
    history = batch_entry.context_as_text[0]
    worst_response = chart_responses[np.argsort(pred_scores_.detach().cpu().numpy())[-1]]
    entries = make_gpt_input_json(history, responses[0], chart_responses[topk_idxs], worst_response)
    entries = dropout_entries(entries,pred_scores)
    gpt_data_ = tokenize_for_gpt(entries, gpt_tokenizer)
    gpt_data = convert_to_gpt_input(gpt_data_, args, gpt_tokenizer)
    gpt_loss_array, mc_loss = train_gpt(gpt_data,gpt,gpt_optimizer,i,args)
    gpt_loss = - gpt_loss_array.sum().detach().item()

    ###GPT DONE###
    combined_loss = scorer.get_combined_loss(pred_scores,gpt_loss_array)
    scorer_optimizer.zero_grad()  
    gpt_optimizer.zero_grad()  
    combined_loss.backward()
    if not args.use_transformer:
        scorer.input_encoder.utterance_encoder.emb_lookup.weight.grad[mask]=0.0 
    scorer_optimizer.step()
    gpt_optimizer.step()

    ranking_info, _, _ = get_ranking_info(gt_labels, pred_scores_)
    
    predicted_scores = (pred_scores).float().detach().cpu().numpy()
    return combined_loss, gpt_loss, ranking_info, [predicted_scores,gpt_loss_array.detach().cpu().numpy()]

def create_output_entry(history,response,output,top5,top5_scores, output_beams, output_probs,startIdx,correct_r1,correct_r5,gpt_loss_array):
    entry = {}
    entry['history']=history
    entry['GT response']=str(response)
    entry['generated response']=output
    entry['Idx']=int(startIdx)
    entry['Correct R1']=bool(correct_r1)
    entry['Correct R5']=bool(correct_r5)

    ranked_topk = []
    top5_scores = top5_scores.cpu().numpy()
    for i, r in enumerate(top5):
        ranked_topk.append({"rank":i+1,"response":r,"probab":float(top5_scores[i]),"gpt ll":float(gpt_loss_array[i])})
    entry['ranked top 5']=ranked_topk

    outputs = []
    sorted_idxs = np.argsort(output_probs)[::-1]
    output_probs = np.array(output_probs)[list(sorted_idxs)]
    output_beams = np.array(output_beams)[list(sorted_idxs)]
    for i, r in enumerate(output_beams):
        outputs.append({"beam":str(r),"probab":float(output_probs[i])})
    entry['Output Beams']=outputs

    return entry

def create_output_entry_new(history,response,output,top5,top5_scores,startIdx,correct_r1,correct_r5,gpt_loss_array):
    entry = {}
    entry['history']=history
    entry['GT response']=str(response)
    entry['generated response']=output
    entry['Idx']=int(startIdx)
    entry['Correct R1']=bool(correct_r1)
    entry['Correct R5']=bool(correct_r5)

    ranked_topk = []
    top5_scores = top5_scores.cpu().numpy()
    for i, r in enumerate(top5):
        ranked_topk.append({"rank":i+1,"response":r,"probab":float(top5_scores[i]),"gpt ll":float(gpt_loss_array[i])})
    entry['ranked top 5']=ranked_topk
    return entry

def validate(data,startIdx, endIdx,sample=False):
    #n_topk = args.scorer_topk
    n_topk = 1
    scorer.eval()
    batch_entry = ProxyScoreBatch(data, glob, startIdx, endIdx)
    contexts,  context_utterance_lengths, context_lengths  = torch.as_tensor(batch_entry.context).long(),  torch.as_tensor(batch_entry.context_utterance_lengths).long(),  torch.as_tensor(batch_entry.context_num_utterances).long()
    paths,  path_utterance_lengths, path_lengths  = torch.as_tensor(batch_entry.path).long(),  torch.as_tensor(batch_entry.path_utterance_lengths).long(),  torch.as_tensor(batch_entry.path_num_utterances).long()
    scores = torch.as_tensor(batch_entry.score,dtype=float)
    responses, chart_responses = batch_entry.response_as_text, batch_entry.chart_response_as_text
    gt_labels = batch_entry.gt_label
    assert len(list(set(responses)))==1 #make sure the batch belongs to the same context-response pair
    
    if args.cuda:
        contexts,  context_utterance_lengths, context_lengths  = contexts.cuda(),  context_utterance_lengths.cuda(), context_lengths.cuda()
        paths, path_utterance_lengths, path_lengths  = paths.cuda(),  path_utterance_lengths.cuda(), path_lengths.cuda()
        scores = scores.cuda()

    with torch.no_grad():
        pred_scores_ = scorer.get_scores(contexts,context_utterance_lengths,context_lengths,paths, path_utterance_lengths, path_lengths)
    #convert to a format such that greater score=better
    topk_idxs = np.argsort(pred_scores_.detach().cpu().numpy())[:args.scorer_topk]#fetch top k
    pred_scores = pred_scores_[list(topk_idxs)]
    pred_scores = softmax(-pred_scores)    
    #GPT#########
    history = batch_entry.context_as_text[0]
    worst_response = chart_responses[np.argsort(pred_scores_.detach().cpu().numpy())[-1]]
    entries = make_gpt_input_json(history, responses[0], chart_responses[topk_idxs], worst_response)
    gpt_data_ = tokenize_for_gpt(entries, gpt_tokenizer)
    gpt_data = convert_to_gpt_input(gpt_data_, args, gpt_tokenizer)
    response_length = torch.sum(gpt_data[2][0]!=-100).item()
    
    gpt_loss_array, (lm_loss_total,n_loss), gpt_loss = gpt_inference(gpt_data,gpt,args)
    _, (_,_), gpt_loss_top = gpt_inference([x[:1] for x in gpt_data],gpt,args)
    #gpt_loss = - gpt_loss_array.sum().item()

    ##SAMPLING#####
    personalities_ = [x['personality'] for x in np.array(gpt_data_)]
    history_ = gpt_data_[0]['utterances'][0]['history']#because history is constant in a batch
    inference_scores = pred_scores[:n_topk]
    personalities_ = personalities_[:n_topk]
    ## gpt_output, output_beams, output_probs = gpt_infer_sequence(personalities_,history_,gpt,gpt_tokenizer,args,inference_scores,test=True)

    if args.inference==0:
        gpt_output = gpt_infer_sequence_new(personalities_,history_,gpt,gpt_tokenizer,args,inference_scores,test=True)
    else:
        gpt_output, output_beams, output_beams_tokens, output_probs = gpt_infer_sequence_beam(personalities_,history_,gpt,gpt_tokenizer,args,inference_scores,test=True)

    ###GPT DONE###
    #Get losses and metrics#
    with torch.no_grad():
        combined_loss = scorer.get_combined_loss(pred_scores,gpt_loss_array)
    ranking_info, correct_r1, correct_r5 = get_ranking_info(gt_labels, pred_scores_)
    if 'last_node' in batch_entry.__dict__ and np.sum(batch_entry.last_node)==1:
        ranking_info=correct_r1=correct_r5=None
    bleu_input = [gpt_output, responses[0]]

    #entry = create_output_entry(history,responses[0],gpt_output,chart_responses[topk_idxs],pred_scores, output_beams, output_probs,startIdx, correct_r1, correct_r5,gpt_loss_array)
    entry = create_output_entry_new(history,responses[0],gpt_output,chart_responses[topk_idxs],pred_scores,startIdx, correct_r1, correct_r5,gpt_loss_array)
    
    avg_combined_loss = combined_loss.item()/response_length
    
    return combined_loss, gpt_loss, ranking_info, bleu_input, entry, (lm_loss_total, n_loss), gpt_loss_top, avg_combined_loss

def calculate_rank_stats(ranking_info):
    correct = [x[1] for x in ranking_info]
    ranked_list = [x[0] for x in ranking_info]
    p_at1 = np.mean([r_at_k([correct[i]],ranked_list[i],1) for i,_ in enumerate(ranked_list)])
    p_at5 = np.mean([r_at_k([correct[i]],ranked_list[i],5) for i,_ in enumerate(ranked_list)])
    #print("r@1:{:.4f}, r@5:{:.4f}".format(p_at1, p_at5))
    return p_at1, p_at5

def r_at_k(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]
    num_hits = 0.0
    for i,p in enumerate(predicted):
        if p in actual:
            num_hits += 1.0
    if not actual:
        return 0.0
    return num_hits / min(len(actual), k)

def initial_rank_stats(batches,data):
    val_ranking_info = []
    bleu_data = []
    entries = []
    lm_loss_total = 0
    n_loss_total = 0
    total_gpt_loss = 0
    total_gpt_loss_top = 0
    total_combined_loss = 0
    for (start_idx, end_idx) in batches:
        combined_loss, gpt_loss, ranking_info_, bleu_input, entry, (lm_loss_, n_loss), gpt_loss_top, avg_combined_loss = validate(data, start_idx, end_idx)
        if ranking_info_!=None:
            val_ranking_info.append(ranking_info_)
        bleu_data.append(bleu_input)
        entries.append(entry)
        lm_loss_total+=lm_loss_
        n_loss_total+=n_loss
        total_gpt_loss+=gpt_loss
        total_gpt_loss_top += gpt_loss_top
        total_combined_loss+=avg_combined_loss

        #print("combined",avg_combined_loss)
        #print("gpt_loss_top",gpt_loss_top)

    val_r_at1, val_r_at5 = calculate_rank_stats(val_ranking_info)
    ppl=np.exp(lm_loss_total/n_loss_total)
    bleu = get_bleu(bleu_data)
    avg_ppl = torch.exp(total_gpt_loss/len(batches)).cpu().numpy()
    avg_ppl_top = torch.exp(total_gpt_loss_top/len(batches)).cpu().numpy()
    comb_ppl = np.exp(total_combined_loss/len(batches))
    return val_r_at1, val_r_at5, bleu, entries, ppl, avg_ppl, comb_ppl, avg_ppl_top

if __name__ == "__main__":
    args = parge_args()
    glob, trnData, valData, trn_batches, val_batches = get_dataset_and_batches()#one dialog utterance against complete chart
    args = set_additional_args(args)#based on dataset
    args, mask = process_glove_matrix(glob,args)

    print("create model")
    sys.stdout.flush()
    scorer, scorer_optimizer, gpt, gpt_optimizer, gpt_tokenizer = init_model(args)

    vcombined_loss_min = 10000000
    vgpt_loss_min = 10000000
    vbleu_max = 0
    val_bleu_max = 0
    combined_losses = []
    r_at_5_array = []
    r_at_1_array = []
    test_r_at_5_array = []
    test_r_at_1_array = []
    gpt_losses = []
    val_bleu_array = []
    test_bleu_array = []
    logger.info('Proxycombined_losses in '+ args.log_dir)
    sys.stdout.flush()

    #getting some initial stats
    print("calculating initial stats:")
    sys.stdout.flush()
    #_, train_r_at5, _, _, _ = initial_rank_stats(trn_batches,trnData)
    #_, val_r_at5, val_bleu, _, _ = initial_rank_stats(val_batches,valData)
    tst_r1, tst_r_at5, tst_bleu, entries, test_ppl, avg_ppl, comb_ppl, avg_ppl_top = initial_rank_stats(val_batches,valData)
    r_at_5_array.append([0, np.nan, np.nan])
    r_at_1_array.append([0, np.nan, np.nan])
    val_bleu_array.append([0,np.nan,np.nan])
    test_bleu_array.append([0,tst_bleu])
    test_r_at_1_array.append([0,tst_r1])  
    test_r_at_5_array.append([0,tst_r_at5])  

    print("Test BLEU:",tst_bleu,", Test R@1:",tst_r1,", Test R@5:",tst_r_at5, "Combined Test Top PPL:", avg_ppl_top)
    with open(args.test_output_path.replace(".json",str(-1)+".json"),"w") as f:
        json.dump(entries,f, indent=4)
    print("start training#########")
    sys.stdout.flush()

    for epoch in range(args.num_epochs):
        t0 = time.time()
        shuffled_trn_batches = copy.deepcopy(trn_batches)
        np.random.shuffle(shuffled_trn_batches)
        combined_loss, vcombined_loss, vtop_loss, vavg_combined_loss = 0, 0, 0, 0
        gpt_loss, vgpt_loss, vgpt_bleu = 0, 0, 0
        train_ranking_info = []
        val_ranking_info = []
        v_bleu_data = []
        scores_array = []
        val_entries, test_entries = [], []
        epoch_train_loss_array = []
        
        for i, (start_idx, end_idx) in enumerate(shuffled_trn_batches):
            combined_loss_, gpt_loss_, ranking_info, scores_ = train(start_idx, end_idx, mask)
            combined_loss += combined_loss_
            gpt_loss += gpt_loss_
            train_ranking_info.append(ranking_info)
            scores_array.append(scores_)
            train_r_at1, train_r_at5 = calculate_rank_stats(train_ranking_info)
            epoch_train_loss_array.append([i+1,combined_loss_,0])

            #if i%1000==0:
            #    print(start_idx, "batch combined loss", combined_loss.item())
            #sys.stdout.flush()
        combined_loss/=len(shuffled_trn_batches)
        gpt_loss/=len(shuffled_trn_batches)
        plot_scores(scores_array,args.loss_plot_save_path.replace("lossPlot","ScoresPlot"),args.loss_path.replace("Net_loss","Scores"))##
        plot_and_save_loss(epoch_train_loss_array,args.loss_plot_save_path.replace("lossPlot","EpochlossPlot"), "combined_epoch_loss")

        #if epoch%2==0:
        scorer.save(args.save_path, scorer_optimizer, args)
        gpt.save_pretrained(args.gpt_save_path)
        
        for (start_idx, end_idx) in val_batches:
            combined_loss_, gpt_loss_, ranking_info_, bleu_input, val_entry, (_, _), gpt_loss_top,avg_combined_loss = validate(valData, start_idx, end_idx)
            vcombined_loss+=combined_loss_
            vavg_combined_loss+=avg_combined_loss
            vgpt_loss+=gpt_loss_
            vtop_loss += gpt_loss_top
            if ranking_info_!=None:
                val_ranking_info.append(ranking_info_)
            v_bleu_data.append(bleu_input)
            val_entries.append(val_entry)
        vcombined_loss/=len(val_batches)
        vavg_combined_loss/=len(val_batches)
        vgpt_loss/=len(val_batches)
        vtop_loss/=len(val_batches)
        val_bleu = get_bleu(v_bleu_data)
        val_bleu_array.append([epoch+1,val_bleu,val_bleu])  
        val_r_at1, val_r_at5 = calculate_rank_stats(val_ranking_info)
        avg_ppl = torch.exp(vgpt_loss).cpu().numpy()
        top_ppl = torch.exp(vtop_loss).cpu().numpy()
        comb_ppl = np.exp(vavg_combined_loss)
        t1 = time.time()
        print("Epoch", (epoch+1), "Loss {0:.3f}".format(combined_loss), "GPT Loss {0:.3f}".format(gpt_loss), "AVG PPL {0:.3f}".format(avg_ppl), "VAL COMBINED PPL {0:.3f}".format(comb_ppl), "VAL TOP PPL {0:.3f}".format(top_ppl), "Vcombined_loss {0:.3f}".format(vcombined_loss), "VGPT_BLEU {0:.2f}".format(val_bleu*100), "Time {0:.3f}".format(t1-t0))
        sys.stdout.flush()
        combined_losses.append([epoch+1, combined_loss.item(), vcombined_loss.item()])
        gpt_losses.append([epoch+1, gpt_loss, vgpt_loss])
        r_at_5_array.append([epoch+1, train_r_at5, val_r_at5])
        r_at_1_array.append([epoch+1, train_r_at1, val_r_at1])
        np.savetxt(args.loss_path, combined_losses, delimiter=",")
        plot_and_save_loss(combined_losses,args.loss_plot_save_path)
        plot_and_save_loss(gpt_losses,args.loss_plot_save_path.replace("lossPlot","GPTlossPlot"), "gpt_loss")
        plot_and_save_loss(r_at_5_array,args.rank_plot_save_path, "r@5",test_r_at_5_array)
        plot_and_save_loss(r_at_1_array,args.rank_plot_save_path.replace("Plot","PlotR1"), "r@1",test_r_at_1_array)
        plot_and_save_loss(val_bleu_array,args.loss_plot_save_path.replace("lossPlot","valBLEUPlot"), "BLEU",test_bleu_array)

        with open(args.val_output_path,"w") as f:
            json.dump(val_entries,f, indent=4)

        if val_bleu > val_bleu_max:
            val_bleu_max = val_bleu
            scorer.save(args.best_path, scorer_optimizer, args)
            gpt.save_pretrained(args.gpt_best_path)
            
            with io.open(args.metric_path, 'w', encoding='utf8') as f:
                f.write("Epoch:{:.0f}, Loss {:.3f}, BLEU {:.3f}, R@1:{:.3f}, R@5:{:.3f}, AVG combined PPL:{:.4f}".format(epoch+1, vcombined_loss, val_bleu*100, val_r_at1, val_r_at5, comb_ppl))
            
            with open(args.test_output_path.replace(".json",str(epoch)+"-val.json"),"w") as f:
                json.dump(test_entries,f, indent=4)
    
    #writer.flush()
    

