# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import json
import os
from os.path import isfile, join
from .Flowcharts import Flowcharts
from random import choices, choice
from copy import deepcopy
import itertools
import pickle
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

PATH_SEPARATOR = " "
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>", "<fsep>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>'], 'flowchart_separator': '<fsep>'}
num_samples = 1
flowchart_separator = " "#ATTR_TO_SPECIAL_TOKEN['flowchart_separator']
tuple_separator = ' '
intra_tuple_separator = ','
tf_idf_path="../data/saved_data/chart_vectorizer.pkl"
root_path="../data/flodial/dataset/"
domain="in_domain_hard"
graph_path='../data/flowcharts/'
num_tf_idf_responses=1
tfidf_separator = " <SEP> "


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

def get_flowchart_data_from_names(dialogs,flowchart_personality,names):
    all_data = []
    agent_dialog_dump = get_agent_dialog_dump(dialogs)
    for name in names:
        print("processing chart:", name)
        f_dialogs_data = get_flowchart_dialogs_as_input(dialogs,flowchart_personality,name,agent_dialog_dump)
        all_data+=f_dialogs_data
        print("\tnum of dialogs:", len(f_dialogs_data))
    return all_data

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

def fetch_using_tf_idf(vectorizer, responses, X, name, query_context,n):
    Y = vectorizer.transform([query_context])

    ranked_docs = get_top_docs(eucledian_distance(Y,X),topk=n,order='ascending')
    retrieved_responses = [responses[x[0]] for x in ranked_docs]
    return retrieved_responses

def get_flowchart_dialogs_as_input_with_tfidf(dialogs,flowchart_jsons,name,agent_dialog_dump, tf_idf_data, n_retrieved):
    #convert into a format suitable for itne input
    agent_dialogs_dump_local = deepcopy(agent_dialog_dump)
    agent_dialogs_dump_local.pop(name)

    vectorizer=tf_idf_data['vectorizer']
    responses=tf_idf_data['responses'][name]
    X=vectorizer.transform(tf_idf_data['contexts'][name])

    flowchart_data_input = []
    for dialog_dict in dialogs[name]['dialogs']:
        dialog = dialog_dict['utterences']
        dialog_utterences = [ process_string(i['utterance']) for i in dialog ]
        for i in range(0,len(dialog),2):
            local_dict = {}
            local_dict['history'] = dialog_utterences[:i+1]
            correct_candidate = dialog_utterences[i+1]
            incorrect_candidates = sample_incorrect_utterances(deepcopy(agent_dialogs_dump_local),name,num_samples)
            local_dict['candidates'] = incorrect_candidates+[correct_candidate]
            personality = fetch_using_tf_idf(vectorizer, responses, X, name,tfidf_separator.join(local_dict['history']),n_retrieved)
            flowchart_data_input.append({'personality':personality,'utterances':[local_dict]})
    return flowchart_data_input

def get_flowchart_data_from_names_with_tfidf(dialogs, flowchart_jsons, names, n_retrieved, tf_idf_path, train=True):
    all_data = []
    agent_dialog_dump = get_agent_dialog_dump(dialogs)
    with open(tf_idf_path,"rb") as f:
        tf_idf_data = pickle.load(f)
    for name in names:
        print("processing chart:", name)
        f_dialogs_data = get_flowchart_dialogs_as_input_with_tfidf(dialogs,flowchart_jsons,name,agent_dialog_dump,tf_idf_data,n_retrieved)
        all_data+=f_dialogs_data
        print("\tnum of dialogs:", len(f_dialogs_data))
    return all_data

def get_flowchart_dialogs_as_input_with_utterance(dialogs,flowchart_jsons,name,agent_dialog_dump):
    #convert into a format suitable for itne input
    agent_dialogs_dump_local = deepcopy(agent_dialog_dump)
    agent_dialogs_dump_local.pop(name)
    flowchart_data_input = []
    for dialog_dict in dialogs[name]['dialogs']:
        dialog = dialog_dict['utterences']
        dialog_utterences = [ process_string(i['utterance']) for i in dialog ]
        dialog_nodes = [ process_string(i['node']) for i in dialog ]
        for i in range(0,len(dialog),2):
            local_dict = {}
            local_dict['history'] = dialog_utterences[:i+1]
            correct_candidate = dialog_utterences[i+1]
            correct_node = dialog_nodes[i+1]
            incorrect_candidates = sample_incorrect_utterances(deepcopy(agent_dialogs_dump_local),name,num_samples)
            local_dict['candidates'] = incorrect_candidates+[correct_candidate]
            personality = [flowchart_jsons[name]['nodes'][correct_node]['utterance']]
            flowchart_data_input.append({'personality':personality,'utterances':[local_dict]})
    return flowchart_data_input

def get_flowchart_data_from_names_with_utterance(dialogs,flowchart_jsons,names, train=True):
    all_data = []
    agent_dialog_dump = get_agent_dialog_dump(dialogs)
    for name in names:
        print("processing chart:", name)
        f_dialogs_data = get_flowchart_dialogs_as_input_with_utterance(dialogs,flowchart_jsons,name,agent_dialog_dump)
        all_data+=f_dialogs_data
        print("\tnum of dialogs:", len(f_dialogs_data))
    return all_data

def process_string(string):
    string=string.strip()
    string=string.replace(".", " .")
    string=string.replace("?", " ?")
    return string

def flowchart_personality_serialized(flowchart_jsons,name,tuple_separator,intra_tuple_separator):
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
def process_flowchart(file_path,flowchart_path):
    files=listdir(file_path)
    flowcharts = {}
    dialogs = {}
    linear_flowchart_paths = {}
    flowchart_personality = {}
    flowchart_jsons = {}

    for file_name in files:
        flowchart_name = file_name.split('.')[0]
        path=flowchart_path+flowchart_name+"/"+file_name
        with open(path,"r") as f1:
            flowchart_jsons[flowchart_name] = json.load(f1)
        with open(file_path+file_name, 'r', encoding="utf-8") as f2:
            dialogs[flowchart_name] = json.load(f2)
        #flowchart_personality[flowchart_name] = get_text_chart_triples(flowchart_jsons[flowchart_name],flowchart_separator)
        flowchart_personality[flowchart_name] = flowchart_personality_serialized(flowchart_jsons,flowchart_name,tuple_separator,intra_tuple_separator)
    return flowcharts, dialogs, linear_flowchart_paths, flowchart_personality, flowchart_jsons

def create_gpt_input(root_path,domain,tf_idf_path,graph_path,save_root,type='tfidf',dummy_data=False):#type='tfidf' or 'dialog'
    flowcharts, dialogs, linear_flowchart_paths, flowchart_personality, flowchart_jsons = process_flowchart(root_path+domain+"/train/",graph_path)
    flowcharts_val, dialogs_val, linear_flowchart_paths_val, flowchart_personality_val, flowchart_jsons_val = process_flowchart(root_path+domain+"/val/",graph_path)
    flowcharts_test, dialogs_test, linear_flowchart_paths_test, flowchart_personality_test, flowchart_jsons_test = process_flowchart(root_path+domain+"/test/",graph_path)

    train_charts = list(flowchart_jsons.keys())
    val_charts = list(flowchart_jsons_val.keys())
    test_charts = list(flowchart_jsons_test.keys())
    trn_empty_personality = dict(zip(train_charts,[[]]*len(train_charts)))
    val_empty_personality = dict(zip(val_charts,[[]]*len(val_charts)))
    tst_empty_personality = dict(zip(test_charts,[[]]*len(test_charts)))

    if type=='tfidf':
        #with flowchart utterances fetched using tfidf as personality - each dialog history has new personality
        train = get_flowchart_data_from_names_with_tfidf(dialogs,flowchart_jsons,train_charts, 1, tf_idf_path, train=True)
        test = get_flowchart_data_from_names_with_tfidf(dialogs_test,flowchart_jsons_test,test_charts, 1, tf_idf_path, train=False)
        val = get_flowchart_data_from_names_with_tfidf(dialogs_val,flowchart_jsons_val,val_charts, 1, tf_idf_path, train=False)
        json_data = {'train':train,'valid':val,'test':test}
        with open(save_root+"/flowchart_data_"+domain+"_tfidf_personality.json",'w') as f:
            json.dump(json_data,f)
    else:
        #with only dialog history
        train = get_flowchart_data_from_names(dialogs,trn_empty_personality,train_charts)
        test = get_flowchart_data_from_names(dialogs_test,tst_empty_personality,test_charts)
        val = get_flowchart_data_from_names(dialogs_val, val_empty_personality,val_charts)
        json_data = {'train':train,'valid':val,'test':test}
        with open(save_root+"/flowchart_"+domain+"_dialog_history.json",'w') as f:
            json.dump(json_data,f)
    
    if dummy_data:
        temp_train=train[:3]
        temp_val=val[:2]
        temp_test=test[:2]
        json_data = {'train':temp_train,'valid':temp_val,'test':temp_test}
        with open(save_root+"/"+type+"_dummy_data"+domain+".json",'w') as f:
            json.dump(json_data,f)

    '''
    #with correct flowchart utterance as personality - each dialog history has new personality
    train = get_flowchart_data_from_names_with_utterance(dialogs,flowchart_jsons,train_charts, train=True)
    test = get_flowchart_data_from_names_with_utterance(dialogs_test,flowchart_jsons_test,test_charts, train=False)
    val = get_flowchart_data_from_names_with_utterance(dialogs_val,flowchart_jsons_val,val_charts, train=False)
    json_data = {'train':train,'valid':val,'test':test}
    with open("./flowchart_data_out_domain_correct_utterance_personality.json",'w') as f:
        json.dump(json_data,f)

    #with flowchart as personality
    train_ = get_flowchart_data_from_names(dialogs,flowchart_personality,train_charts, train=True)
    test_ = get_flowchart_data_from_names(dialogs_test,flowchart_personality_test,test_charts, train=False)
    val_ = get_flowchart_data_from_names(dialogs_val,flowchart_personality_val,val_charts, train=False)

    json_data = {'train':train_,'valid':val_,'test':test_}
    with open("./flowchart_data_personality_w_test.json",'w') as f:
        json.dump(json_data,f)
    '''
    '''
    # %%
    temp_train=train[:3]
    temp_val=val[:2]
    temp_test=test[:2]
    json_data = {'train':temp_train,'valid':temp_val,'test':temp_test}
    with open("./flowchart_data_in_domain_tfidf1_personality_SMALL.json",'w') as f:
        json.dump(json_data,f)

    temp_train=train_[:3]
    temp_val=val_[:2]
    temp_test=test_[:2]
    json_data = {'train':temp_train,'valid':temp_val,'test':temp_test}
    with open("./flowchart_data_personality_w_test_SMALL.json",'w') as f:
        json.dump(json_data,f)
    '''
