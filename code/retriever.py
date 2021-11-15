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
import json
import argparse
from Model.ProxyScore import ProxyScore
import nltk
import pickle
from utils.Flowcharts import Flowcharts
from utils.ProxyScoreData import ProxyScoreData, ProxyScoreBatch
from utils.proxy_scores import get_scores_dict
#from torch.utils.tensorboard import SummaryWriter
from utils import read_flowchart_jsons, read_dialog_jsons, build_vocab, create_batches, PAD, get_indexes_for_bleu, read_flowchart_doc_jsons, PAD_INDEX, CLS_INDEX, cache_embedding_matrix
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate FlowNet')

    parser.add_argument('--flowchart-dir', type=str, default='../data/flowcharts/', help='a directiory that contains all flowcharts')
    parser.add_argument('--model_checkpoint', type=str, default='', help='a directiory that contains all flowcharts')

    #change these depending on the dataset
    parser.add_argument('--cached-dialog-path', type=str, default='../data/saved_data/cached_in_domain_hard_dialogs.pkl', help='cached dataset path')
    parser.add_argument('--domain', type=str, default='in_domain_hard', help='in_domain_hard, out_domain')
    parser.add_argument('--save-name', type=str, default="TEST", help='Name of model to be saved')

    parser.add_argument('--dialog-dir', type=str, default='../data/new_dialogs/', help='a directiory that contains trn, val and tst dialogs')
    parser.add_argument('--cached-scores-path', type=str, default='../data/saved_data/', help='cached dataset path')
    parser.add_argument('--gpt_data_save_path', type=str, default='../data/gpt_data/', help='cached dataset path')
    parser.add_argument('--saved-glove-path', type=str, default='./glove6B/', help='cached embedding matrix with glove embeddings')
    parser.add_argument('--n_heads', type=int, default=2)

    parser.add_argument('--model-dir', type=str, default='../data/model/', help='dataset')
    parser.add_argument('--log-dir', type=str, default='../logs/', help='save logs of the runs')

    parser.add_argument('--best', action='store_true', default=False, help='Load the best model so far')
    parser.add_argument('--load-name', type=str, default="", help='Name of model to be loaded')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume an old model (default: False)')
    parser.add_argument('--save-step', type=int, default=2000, metavar='SS', help='how many batches to wait before saving model')
    
    parser.add_argument('--num-epochs', type=int, default=1000, metavar='E', help='Number of epochs for training the model')
    parser.add_argument('--max-steps', type=int, default=200000, help='Max steps per epoch')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 10)')
    parser.add_argument('--lr', type=float, default=.0001, metavar='LR', help='Learning Rate (default: .00015)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='LR', help='Learning Rate (default: .00015)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--val-step', type=int, default=1, metavar='ES', help='how many batches to wait before evaluating 1 test batch')
    parser.add_argument('--log-step', type=int, default=100, metavar='LS', help='how many batches to wait before logging training status')

    parser.add_argument('--margin', type=int, default=2, metavar='H', help='margin for loss training')
    parser.add_argument('--hidden-size', type=int, default=300, metavar='H', help='Size of hidden embeddings')
    parser.add_argument('--emb-size', type=int, default=100, metavar='H', help='Size of hidden embeddings')
    parser.add_argument('--encoder_num_layers', type=int, default=1, metavar='H', help='number of layers in encoder')
    parser.add_argument('--bidirectional-encoder', type=bool, default=True, metavar='H', help='bidirectional encoder')

    args = parser.parse_args()
    if args.save_name == "":
        args.save_name = args.load_name
        
    if args.load_name == "":
        args.load_name = args.save_name

    args.cuda = torch.cuda.is_available()
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    if not os.path.exists(args.gpt_data_save_path):
        os.mkdir(args.gpt_data_save_path)
    if not os.path.exists(args.cached_scores_path):
        os.mkdir(args.cached_scores_path)

    domain_str = args.domain

    params = [args.save_name, domain_str, str(args.hidden_size), str(args.lr), str(args.batch_size)]
    param_str = "_".join(params)
    print("Parameter String is :", param_str)
    args.load_path = args.model_dir + "Proxycheckpoint_" + args.load_name + "_" + domain_str + "_" + "_" + str(args.hidden_size)  + "_" + str(args.lr) + "_" + str(args.batch_size) + '.pth.tar'
    args.save_path = args.model_dir + "Proxycheckpoint_" + param_str + '.pth.tar'
    args.best_path = args.model_dir + "Proxybest_checkpoint_" + param_str + '.pth.tar'
    args.output_path = args.log_dir + "Proxyoutput_" + param_str + '.txt'
    args.metric_path = args.log_dir + "Proxymetric_" + param_str + '.txt'
    args.loss_path = args.log_dir + "Proxyloss_" + param_str + '.txt'
    args.loss_plot_save_path = args.log_dir + "lossPlot_" + param_str + '.png'
    args.ranks_data_save_path = args.cached_scores_path + "../ranks/ranks_" + param_str + '.json'

    args.dialog_dataset_dir = args.dialog_dir + "dataset/" + args.domain + "/"
    args.dialog_dir = args.dialog_dir + args.domain + "/"
    args.cached_scores_path = args.cached_scores_path + "scored_paths_" + args.domain + "_FAQ.json"
    args.gpt_data_save_path = args.gpt_data_save_path + "/Scorer_" + args.save_name + "_" + args.domain + ".json"
    #args.saved_glove_path = args.saved_glove_path + "saved_embedding_matrix" + str(args.emb_size) +"_" + args.domain + "_dialogs_.pkl"
    
    #for siamese
    args.encoder_vocab_size=len(glob['encoder_vocab_to_idx'])
    args.encoder_hidden_size=args.hidden_size
    args.encoder_bidirectional_flag=args.bidirectional_encoder

    args.no_sample = False
    return args

def load_data(args):
    print("loading dataset")
    if os.path.exists(args.cached_dialog_path):
        with open(args.cached_dialog_path,"rb") as f:
            trnData, valData, tstData, glob = pickle.load(f)
    else:
        flowchartsJson = read_flowchart_jsons(args.flowchart_dir)
        flowchartDocsJson = read_flowchart_doc_jsons(args.flowchart_dir)
        trnJson, valJson, tstJson = read_dialog_jsons(args.dialog_dir)
        glob = build_vocab(flowchartsJson, trnJson, valJson)
        flowcharts = Flowcharts(flowchartsJson, glob)
        scores_dict = get_scores_dict(args.dialog_dataset_dir,args.cached_scores_path,args.flowchart_dir)
        trnData = ProxyScoreData(trnJson, flowcharts, glob, scores_dict['train'],flowchartDocsJson,test=True)
        valData = ProxyScoreData(valJson, flowcharts, glob, scores_dict['valid'], flowchartDocsJson,test=True)
        tstData = ProxyScoreData(tstJson, flowcharts, glob, scores_dict['test'], flowchartDocsJson,test=True)
        with open(args.cached_dialog_path,"wb") as f:
            pickle.dump([trnData,valData,tstData,glob],f)

    glove_matrix_path, missed_idx_path = cache_embedding_matrix(glob,args.emb_size,args.saved_glove_path,args.domain+"_dialogs",args.cached_dialog_path)
    args.saved_glove_path = glove_matrix_path
    #  mask the values in vocab (and their embedding matrix indexes)
    #  that are not in the glove embedding matrix
    with open(missed_idx_path,'rb') as f:
        training_idxs, words = pickle.load(f) 
        mask = np.ones(len(glob['encoder_vocab_to_idx'])).astype(bool)
        mask[training_idxs]=False

    print("batching dataset")
    trn_batches = get_indexes_for_bleu(trnData.batche_start)
    val_batches = get_indexes_for_bleu(valData.batche_start)
    tst_batches = get_indexes_for_bleu(tstData.batche_start)

    print("loading dataset complete")
    return trnData, valData, tstData, glob, trn_batches, val_batches, tst_batches, mask

def plot_and_save_loss(losses,args):
    plt.clf()
    losses_ = np.array(losses)
    plt.plot(losses_[:,0], losses_[:,1], label='Train')
    plt.plot(losses_[:,0], losses_[:,2], label='Val')
    #plt.plot(losses[:,0], losses[:,3], label='Test')
    plt.title('Losses')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(args.loss_plot_save_path)

def output_to_utterance(sample,glob):
    output_vocab=glob['decoder_idx_to_vocab']
    output_utterance=[output_vocab[x] for x in sample]
    output_utterance=list(filter((PAD).__ne__,output_utterance))#remove padding 
    return output_utterance

def get_bleu(hypothesis,references):
    hypothesis=[x.split() for x in hypothesis]
    references = [[x.split()] for x in references]
    bleu=nltk.translate.bleu_score.corpus_bleu(references, hypothesis)
    #remove padding 
    return bleu

def init_model(args):
    model = ProxyScore(args, glob)
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    if args.model_checkpoint!="":
        checkpoint = torch.load(args.model_checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    return model, optimizer

def flatten_context( context, lens):
        context_list = []
        speakers_list = []
        positions_list = []
        max_len = 0
        for k in range(context.shape[0]):
            new_context = [CLS_INDEX]
            speakers = [0]
            positions = []
            for i,x in enumerate(context[k]):
                new_context+=list(x[:lens[k][i]])
            for i,x in enumerate(lens[k]):
                speakers += [0]*x if i%2==0 else [1]*x
            positions = list(range(1,len(new_context)+1))
            max_len = max(max_len,len(new_context))
            context_list.append(new_context)
            speakers_list.append(speakers)
            positions_list.append(positions)
        #pad these
        for k in range(context.shape[0]):
            padding = [PAD_INDEX]*(max_len-len(positions_list[k]))
            context_list[k]+=padding
            speakers_list[k]+=padding
            positions_list[k]+=padding
        context_list = torch.tensor(context_list).long()
        speakers_list = torch.tensor(speakers_list).long()
        positions_list = torch.tensor(positions_list).long()
        return context_list, speakers_list, positions_list

def train(startIdx, endIdx):
    
    model.train()
    
    batch_entry = ProxyScoreBatch(trnData, glob, startIdx, endIdx)
    contexts,  context_utterance_lengths, context_lengths  = torch.as_tensor(batch_entry.context).long(),  torch.as_tensor(batch_entry.context_utterance_lengths).long(), torch.as_tensor(batch_entry.context_num_utterances) #assuming context is same for one batch
    paths,  path_utterance_lengths, path_lengths  = torch.as_tensor(batch_entry.path).long(),  torch.as_tensor(batch_entry.path_utterance_lengths).long(),  torch.as_tensor(batch_entry.path_num_utterances).long()
    scores = torch.as_tensor(batch_entry.score,dtype=float)
    responses, chart_responses = batch_entry.response_as_text, batch_entry.chart_response_as_text
    assert len(list(set(responses))) == 1
    if args.cuda:
        contexts,  context_utterance_lengths, context_lengths  = contexts.cuda(),  context_utterance_lengths.cuda(), context_lengths.cuda()
        paths, path_utterance_lengths, path_lengths  = paths.cuda(),  path_utterance_lengths.cuda(), path_lengths.cuda()
        scores = scores.cuda()
        
    loss, distance = model(contexts,context_utterance_lengths,context_lengths,paths, path_utterance_lengths, path_lengths, scores)

    optimizer.zero_grad()  
    loss.backward() 
    model.input_encoder.utterance_encoder.emb_lookup.weight.grad[mask]=0.0
    optimizer.step()
    
    return loss.item()

def validate(startIdx, endIdx,sample=False):
    model.eval()

    batch_entry = ProxyScoreBatch(valData, glob, startIdx, endIdx)
    contexts,  context_utterance_lengths, context_lengths  = torch.as_tensor(batch_entry.context).long(),  torch.as_tensor(batch_entry.context_utterance_lengths).long(), torch.as_tensor(batch_entry.context_num_utterances) #assuming context is same for one batch
    paths,  path_utterance_lengths, path_lengths  = torch.as_tensor(batch_entry.path).long(),  torch.as_tensor(batch_entry.path_utterance_lengths).long(),  torch.as_tensor(batch_entry.path_num_utterances).long()
    scores = torch.as_tensor(batch_entry.score,dtype=float)
    responses, chart_responses = batch_entry.response_as_text, batch_entry.chart_response_as_text
    assert len(list(set(responses))) == 1
    if args.cuda:
        contexts,  context_utterance_lengths, context_lengths  = contexts.cuda(),  context_utterance_lengths.cuda(), context_lengths.cuda()
        paths, path_utterance_lengths, path_lengths  = paths.cuda(),  path_utterance_lengths.cuda(), path_lengths.cuda()
        scores = scores.cuda()
    with torch.no_grad():    
        loss, distance = model(contexts,context_utterance_lengths,context_lengths,paths, path_utterance_lengths, path_lengths, scores)

    return loss.item()

def test(startIdx, endIdx, sample=False):
    model.eval()
    batch_entry = ProxyScoreBatch(tstData, glob, startIdx, endIdx)
    contexts,  context_utterance_lengths, context_lengths  = torch.as_tensor(batch_entry.context).long(),  torch.as_tensor(batch_entry.context_utterance_lengths).long(), torch.as_tensor(batch_entry.context_num_utterances) #assuming context is same for one batch
    paths,  path_utterance_lengths, path_lengths  = torch.as_tensor(batch_entry.path).long(),  torch.as_tensor(batch_entry.path_utterance_lengths).long(),  torch.as_tensor(batch_entry.path_num_utterances).long()
    scores = torch.as_tensor(batch_entry.score,dtype=float)
    responses, chart_responses = batch_entry.response_as_text, batch_entry.chart_response_as_text
    assert len(list(set(responses))) == 1
    if args.cuda:
        contexts,  context_utterance_lengths, context_lengths  = contexts.cuda(),  context_utterance_lengths.cuda(), context_lengths.cuda()
        paths, path_utterance_lengths, path_lengths  = paths.cuda(),  path_utterance_lengths.cuda(), path_lengths.cuda()
        scores = scores.cuda()
    with torch.no_grad():    
        loss, calc_scores = model(contexts,context_utterance_lengths,context_lengths,paths, path_utterance_lengths, path_lengths, scores)

    best = np.argsort(calc_scores.cpu().numpy())[0]
    best_response = chart_responses[best]
    reference = list(set(responses))
    assert len(reference)==1

    if np.sum(batch_entry.gt_label)==0:
        gt_index, ranked_idxs = 1, [0,0,0,0,0,0,0,0,0,0,0,1]
    else:
        gt_labels = batch_entry.gt_label
        gt_indexes = np.where(np.array(gt_labels)==1)[0]
        ranked_idxs = np.argsort(calc_scores.cpu().numpy())#flip in case of scores
        gt_index = [x for x in ranked_idxs if x in gt_indexes][0]
    
    if 'last_node' in batch_entry.__dict__ and np.sum(batch_entry.last_node)==1:
        gt_index, ranked_idxs=None, None

    return loss.item(), (best_response, reference[0]), (gt_index, ranked_idxs)

def calculate_rank_stats(correct,ranked_list):
    p_at1 = np.mean([r_at_k([correct[i]],ranked_list[i],1) for i,_ in enumerate(ranked_list)])
    p_at5 = np.mean([r_at_k([correct[i]],ranked_list[i],5) for i,_ in enumerate(ranked_list)])
    print("r@1:{:.4f}, r@5:{:.4f}".format(p_at1, p_at5))
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

###GPT input creation
def process_string(string):
    string=string.strip()
    string=string.replace(".", " .")
    string=string.replace("?", " ?")
    return string

def choose_negative_candidate(history,worst_response):
    #pick agent utterances
    agent_utts = history[1::2]
    negative_candidate = ""
    if random.uniform(0, 1)>0.5 or len(agent_utts)==0:
        negative_candidate = worst_response
    else:
        negative_candidate = random.choice(agent_utts)
    return negative_candidate

def create_entry(Data, glob, startIdx, endIdx, best_response, Test=False):
    batch_entry = ProxyScoreBatch(Data, glob, startIdx, endIdx)
    #make sure batching is correct
    reference = list(set(batch_entry.response_as_text))
    assert len(reference)==1
    #retrieve text entries
    history = batch_entry.context_as_text[0]#since all will be same within a batch
    response = batch_entry.response_as_text[0]#since all will be same within a batch
    if not Test:
        best_response = batch_entry.chart_response_as_text[np.where(batch_entry.label==1)[0][0]]
    worst_response = np.random.choice([x for x in batch_entry.chart_response_as_text if x!=best_response])

    negative_candidate = choose_negative_candidate(history,worst_response)
    
    #process everything (spaced . and ?)
    best_response = process_string(best_response)
    negative_candidate = process_string(negative_candidate)
    response = process_string(response)
    history = [process_string(x) for x in history]
    
    entry = {"personality": [best_response],
             "utterances":[{"candidates":[negative_candidate,response],
                 "history":history}]
            }
    return entry

def create_output_entry(Data, glob, startIdx, endIdx, best_response, Test=False):
    batch_entry = ProxyScoreBatch(Data, glob, startIdx, endIdx)
    history = batch_entry.context_as_text[0]#since all will be same within a batch
    response = batch_entry.response_as_text[0]#since all will be same within a batch
    
    #process everything (spaced . and ?)
    best_response = process_string(best_response)
    response = process_string(response)
    history = [process_string(x) for x in history]
    
    entry = {"Fetched Response": [best_response],
             "GT Response":response,
             "history":history
            }
    return entry
 
args = parse_args()
print(args)
trnData, valData, tstData, glob, trn_batches, val_batches, tst_batches, mask = load_data(args)

print("create model")
#torch.manual_seed(6005537891671269197)
print("INITIAL SEED:",torch.initial_seed())
model, optimizer = init_model(args)

vloss_min = 10000000
vbleu_max = 0
losses = []
#writer = SummaryWriter(log_dir=args.log_dir,filename_suffix='ProxyLosses')
print("CUDA", args.cuda)
print("Loaded Data Files")
sys.stdout.flush()
print("start training#########")
train_entries = []
val_entries = []
make_train_entries = make_val_entries = True
for epoch in range(args.num_epochs):
    
    print("")
    t0 = time.time()

    shuffled_trn_batches = copy.deepcopy(trn_batches)
    np.random.shuffle(shuffled_trn_batches)
    loss = 0
    vloss = 0
    vbleu = 0

    for (start_idx, end_idx) in shuffled_trn_batches:
        #print("starting train:"+start_idx)
        s_loss = train(start_idx, end_idx)
        loss += s_loss
        if make_train_entries:
            train_entries.append(create_entry(trnData, glob, start_idx, end_idx, ""))
    if loss == np.nan:
        break
    loss/=len(shuffled_trn_batches)
    model.save(args.save_path, optimizer, args)
    make_train_entries = False
    #writer.add_scalar("Loss/train", loss, epoch)
    
    for (start_idx, end_idx) in val_batches:
        vloss += validate(start_idx, end_idx)
        if make_val_entries:
            val_entries.append(create_entry(valData, glob, start_idx, end_idx, ""))
    vloss/=len(val_batches)
    make_val_entries = False
    t1 = time.time()
    print("Epoch", (epoch+1), "Loss {0:.3f}".format(loss), "VLoss {0:.3f}".format(vloss),"Time {0:.3f}".format(t1-t0))
    sys.stdout.flush()
    losses.append([epoch, loss, vloss])
    plot_and_save_loss(losses,args)

    tloss = 0
    if vloss < vloss_min:
        vloss_min = vloss
        model.save(args.best_path, optimizer, args)
        
        t0 = time.time()
        test_loss = 0
        hypothesis = []
        references = []
        correct_idxs = []
        ranked_lists = []
        test_entries = []
        test_output_entries = []
        for (start_idx, end_idx) in tst_batches:
            loss_, (hypothesis_, refrence_), (correct, ranked_list) = test(start_idx, end_idx,True)
            test_loss+=loss_
            hypothesis.append(hypothesis_)
            references.append(refrence_)
            if correct!=None:
                correct_idxs.append(int(correct))
                ranked_lists.append(list(map(int,ranked_list)))
            test_entries.append(create_entry(tstData, glob, start_idx, end_idx, hypothesis_, Test= True))
            test_output_entries.append(create_output_entry(tstData, glob, start_idx, end_idx, hypothesis_, Test= True))
        test_loss/=len(tst_batches)
        test_bleu = get_bleu(hypothesis,references)   
        t1 = time.time()
        rat1, rat5 = calculate_rank_stats(correct_idxs,ranked_lists)
        print('Test Scores:', "Loss:{0:.3f}".format(test_loss), "BLEU:{0:.4f}".format(test_bleu), "R@1:{0:.4f}".format(rat1), "R@5:{0:.4f}".format(rat5), "Time:{0:.3f}".format(t1-t0))
        sys.stdout.flush()
        
        with io.open(args.metric_path, 'w', encoding='utf8') as f:
            f.write("Train Loss:{:.5f}, Val Loss:{:.5f}, Test Loss:{:.5f}, Test BLEU:{:.5f}, R@1:{:.3f}, R@5:{:.3f}".format(loss,vloss_min,test_loss,test_bleu, rat1, rat5))
         
        Dataset = {"train":train_entries,"valid":val_entries,"test":test_entries}
        with open(args.gpt_data_save_path,'w') as f:
            json.dump(Dataset,f)
        
        Dataset = {"train":[],"valid":[],"test":test_output_entries}
        with open(args.gpt_data_save_path.replace(".json","_test_data.json"),'w') as f:
            json.dump(Dataset,f)

#writer.flush()