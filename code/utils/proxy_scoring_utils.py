
import json
import os
from os.path import isfile, join
from .Flowcharts import Flowcharts
from random import choices, choice
from copy import deepcopy
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from utils import SEPARATOR
import nltk
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu


PATH_SEPARATOR = " " + SEPARATOR + " "

# %%
def get_linear_flowchart_paths(flowchart):
    #convert flowchart into set of paths to each terminal
    paths = flowchart.get_paths_to_all_terminal_nodes()
    linear_paths = []
    for path in paths:
        linear_path = ""
        for node, edge in path:
            if len(linear_path) > 0:
                linear_path += PATH_SEPARATOR
            linear_path += process_string(flowchart.get_utterance(node)) + PATH_SEPARATOR + edge
        linear_paths.append(linear_path)
    return linear_paths

def get_node_utterance(chart, node):
    return process_string(chart['nodes'][node]['utterance'])

def get_all_tuples(chart):
    #convert flowchart into triples, (parent, child, edge)
    tuples = []
    for node, edges in chart['edges'].items():
        for edge, child in edges.items():
            tuples.append((get_node_utterance(chart,node),get_node_utterance(chart,child),edge))
    return tuples

def get_text_chart_triples(chart,sep):
    #convert flowchart into triples, which are linearised as text
    tuples = []
    for node, edges in chart['edges'].items():
        for edge, child in edges.items():
            tuples.append(get_node_utterance(chart,node)+sep+get_node_utterance(chart,child)+sep+edge)
    return tuples

def sample_incorrect_utterances(agent_dialogs_dump_local,flow_name,num_samples):
    #sample a set of utterances
    candidate_utterances=list(itertools.chain(*agent_dialogs_dump_local.values()))
    incorrect_candidates = choices(candidate_utterances,k=num_samples)
    return incorrect_candidates

def get_flowchart_dialogs_as_input(dialogs,flowchart_personality,name,agent_dialog_dump):
    #convert into a format suitable for itne input
    agent_dialogs_dump_local = deepcopy(agent_dialog_dump)
    agent_dialogs_dump_local.pop(name)
    flowchart_data_input = []
    for dialog_dict in dialogs[name]['dialogs']:
        dialog = dialog_dict['utterences']
        dialog_utterences = [ process_string(i['utterance']) for i in dialog ]
        utterances_data = []
        for i in range(0,len(dialog),2):
            local_dict = {}
            local_dict['history'] = dialog_utterences[:i+1]
            correct_candidate = dialog_utterences[i+1]
            incorrect_candidates = sample_incorrect_utterances(deepcopy(agent_dialogs_dump_local),name,num_samples)
            local_dict['candidates'] = incorrect_candidates+[correct_candidate]
            utterances_data.append(local_dict)
        personality = flowchart_personality[name]
        flowchart_data_input.append({'personality':personality,'utterances':utterances_data})
    return flowchart_data_input

def get_flowchart_data_from_names(dialogs,flowchart_personality,names, train=True):
    all_data = []
    agent_dialog_dump = get_agent_dialog_dump(dialogs)
    for name in names:
        print("processing chart:", name)
        f_dialogs_data = get_flowchart_dialogs_as_input(dialogs,flowchart_personality,name,agent_dialog_dump)
        all_data+=f_dialogs_data
        print("\tnum of dialogs:", len(f_dialogs_data))
    return all_data

def process_string(string):
    string=string.strip()
    string=string.replace(".", " .")
    string=string.replace("?", " ?")
    return string

def flowchart_personality_serialized(flowchart_jsons,name,tuple_separator="",intra_tuple_separator=""):
    chart=flowchart_jsons[name]
    tuples = []

    for node, edges in chart['edges'].items():
        for edge, child in edges.items():
            tuples.append(process_string(node)+ "<sep1>" + process_string(child) + "<sep2>" + process_string(edge))
    nodes = [process_string(k)+"<kvsep>"+process_string(v['utterance']) for k,v in chart['nodes'].items()]
    tuples = "<pth>"+"<sep3>".join(tuples)+"<eopth>"
    nodes = "<utt>"+"<usep>".join(nodes)+"<eoutt>"
    return ["<flw>"+tuples+nodes+"<eoflw>"]

def get_agent_dialog_dump(dialogs):
    agnt_dialogs_dump = {}
    for name in list(dialogs.keys()):
        dialog_data = dialogs[name]['dialogs']
        dialog_data = [x['utterences'][:-1] for x in dialog_data]
        dialog_data = itertools.chain(*dialog_data)
        dialog_data = [process_string(x['utterance']) for x in dialog_data if x['speaker']=='agnt']
        agnt_dialogs_dump[name] = dialog_data
    return agnt_dialogs_dump


# %%
from os import listdir
def process_flowchart(file_path,flowchart_path='./data/flowcharts/'):
    files=listdir(file_path)
    flowcharts = {}
    dialogs = {}
    linear_flowchart_paths = {}
    flowchart_personality = {}
    flowchart_jsons = {}
    docs_jsons = {}

    for file_name in files:
        if '.json' not in file_name:
            continue
        flowchart_name = file_name.split('.')[0]
        path=flowchart_path+flowchart_name+"/"+file_name
        doc_path=flowchart_path+flowchart_name+"/"+'supporting_docs.json'
        with open(path,"r") as f1:
            flowchart_jsons[flowchart_name] = json.load(f1)
        with open(file_path+file_name, 'r', encoding="utf-8") as f2:
            dialogs[flowchart_name] = json.load(f2)
        with open(doc_path, 'r', encoding="utf-8") as f3:
            docs_jsons[flowchart_name] = json.load(f3)
        flowchart_personality[flowchart_name] = flowchart_personality_serialized(flowchart_jsons,flowchart_name)
    return flowcharts, dialogs, linear_flowchart_paths, flowchart_personality, flowchart_jsons, docs_jsons

def cosine_similarity(mat1,mat2):
    ##### input is #####
    # mat1 = (#query documents, tf-idf dimensions)
    # mat2 = (#database documents, tf-idf dimensions) 
    n1 = np.linalg.norm(mat1.toarray(),2,axis=-1) 
    n2 = np.linalg.norm(mat2.toarray(),2,axis=-1)   

    return np.dot(mat1.toarray(),mat2.toarray().transpose())/n1[:,None]/n2[None,:]

def cosine_similarity2(mat1,mat2):
    ##### input is #####
    # mat1 = (#query documents, tf-idf dimensions)
    # mat2 = (#database documents, tf-idf dimensions) 
    n1 = np.linalg.norm(mat1,2,axis=-1) 
    n2 = np.linalg.norm(mat2,2,axis=-1)   
   
    return np.dot(mat1,mat2.transpose())/n1[:,None]/n2[None,:]

def eucledian_distance(mat1,mat2):
    ##### input is #####
    # mat1 = (#query documents, tf-idf dimensions)
    # mat2 = (#database documents, tf-idf dimensions)    
    return euclidean_distances(mat1, mat2)

def get_top_docs(scores,topk=5,order='descending'):
    ##### Input #####
    # scores = (#query documents, #database documents)
    #### Output #####
    # ranked_docs = (#query documents, #topk ranked indices in descending order of score)
    order = -1 if order=='descending' else 1
    return np.argsort(scores)[:,::order][:,:topk]

def get_bleu(reference,hypothesis):
    bleu=nltk.translate.bleu_score.corpus_bleu([[x.split()] for x in reference], [x.split() for x in hypothesis])
    return bleu  

##p_at_k variable was changed to store recall at k
def get_documents_from_flowchart(flows,chart):
    paths = flows.get_all_paths(chart)
    chart_contexts = []
    chart_responses = []
    chart_response2node = {}
    for node,path in paths.items():
        text_path = ""
        for n,e in path:
            text_path+=flows.get_node_text(chart, n)+PATH_SEPARATOR+e+PATH_SEPARATOR
        chart_contexts.append(text_path.strip())
        chart_responses.append(flows.get_node_text(chart, node).strip())
        chart_response2node[len(chart_responses)-1]=node
    return chart_contexts, chart_responses, chart_response2node
    
def get_processed_input_data(dialogs_test,chart):
    input_contexts = []
    input_responses = []
    input_labels = []
    chart_dialogs = dialogs_test[chart]['dialogs']
    for chart_dialog in chart_dialogs:
        chart_dialog = chart_dialog['utterences']
        dialog_utterances = [x['utterance'] for x in chart_dialog]
        dialog_node_label = [x['node'] if 'node' in x else None for x in chart_dialog]
        input_contexts+= [PATH_SEPARATOR.join(dialog_utterances[:i]) for i in range(1,len(dialog_utterances),2)]
        input_responses+= [dialog_utterances[i] for i in range(1,len(dialog_utterances),2)]
        input_labels+= [dialog_node_label[i] for i in range(1,len(dialog_utterances),2)]
    return input_contexts, input_responses, input_labels

def make_df_of_tfidf(X, vectorizer):
    feature_names = vectorizer.get_feature_names()
    dense = X.todense()
    denselist = dense.tolist()
    df_X = pd.DataFrame(denselist, columns=feature_names)
    return df_X

def get_reciprocal_ranks(input_label_responses, ranked_docs):
    ranks = [np.where(x==input_label_responses[i])[0][0]+1 for i,x in enumerate(ranked_docs)]
    reciprocal_ranks = [1/x for x in ranks]
    return reciprocal_ranks

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

def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def ark(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    if not actual:
        return 0.0
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / len(actual)
    return score / min(len(predicted), k)

def mark(actual, predicted, k=10):
    return np.mean([ark(a,p,k) for a,p in zip(actual, predicted)])

def get_documents_from_flowchart1(flows,chart):
    paths = flows.get_all_paths(chart)
    chart_contexts = []
    chart_contexts_array = []
    chart_responses = []
    chart_response2node = {}
    for node,path in paths.items():
        text_path = ""
        path_array = []
        for n,e in path:
            text_path+=flows.get_node_text(chart, n)+PATH_SEPARATOR+e+PATH_SEPARATOR
            path_array+=[flows.get_node_text(chart, n),e]
        chart_contexts.append(text_path.strip())
        chart_contexts_array.append(path_array)
        chart_responses.append(flows.get_node_text(chart, node).strip())
        chart_response2node[len(chart_responses)-1]=chart+":"+node
    return chart_contexts, chart_contexts_array, chart_responses, chart_response2node

def get_bleu_matrix(input_responses, chart_responses):
    bleu_mat = np.zeros((len(input_responses),len(chart_responses)))
    for i, resp in enumerate(input_responses):
        for j, chart_resp in enumerate(chart_responses):
            bleu_mat[i][j] = sentence_bleu([resp.split()], chart_resp.split())
    return bleu_mat

def get_documents_from_flowchart2(flows,chart,docs):
    paths = flows.get_all_paths(chart)
    chart_contexts = []
    chart_contexts_array = []
    chart_responses = []
    chart_response2node = {}
    doc_response2node = {}
    chart_docs = []
    for node,path in paths.items():
        text_path = ""
        path_array = []
        for n,e in path:
            text_path+=flows.get_node_text(chart, n)+PATH_SEPARATOR+e+PATH_SEPARATOR
            path_array+=[flows.get_node_text(chart, n),e]
        chart_contexts.append(text_path.strip())
        chart_contexts_array.append(path_array)
        chart_responses.append(flows.get_node_text(chart, node).strip())
        chart_response2node[len(chart_responses)-1]=chart+":"+node
    
    inv_chart_response2node = {v:k for k,v in chart_response2node.items()}
    for d in docs['supporting_faqs']:
        chart_contexts.append(d['q'].strip())
        chart_contexts_array.append([d['q'].strip()])
        chart_responses.append(d['a'].strip())
        doc_response2node[len(chart_responses)-1]=int(d['id'])
    return chart_contexts, chart_contexts_array, chart_responses, chart_response2node, doc_response2node
