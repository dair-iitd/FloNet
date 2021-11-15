#!/usr/bin/env python
# coding: utf-8

# In[116]:


import json
import os
from os.path import isfile, join
from .Flowcharts import Flowcharts
from random import choices, choice
from copy import deepcopy
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import nltk
import pandas as pd
import numpy as np
from .proxy_scoring_utils import process_flowchart, cosine_similarity, get_bleu_matrix, get_processed_input_data, eucledian_distance, get_documents_from_flowchart1, get_top_docs, get_documents_from_flowchart2, r_at_k

sim='euc'
domain='in'
metric = 'tfidf'#[tfidf,bleu,tfidf_bleu]
alphas={'tfidf':0,'bleu':1,'tfidf_bleu':0.1}
alpha = alphas[metric]#optimum alpha=0.1,tfidf=0,bleu=1
eps = 1e-9
def get_scores_dict(dialogs_path="../data/dialogs/dataset/"+domain+"_domain/",save_path='../data/saved_data/scored_paths_'+domain+'_domain.json', flowchart_path= "../data/flowcharts/",resave=False):
    if os.path.isfile(save_path) and not resave:
        print("loading scores dictionary from ",save_path)
        with open(save_path,'r') as f:
            return json.load(f)

    _, dialogs, _, _, flowchart_jsons, docs_jsons = process_flowchart(dialogs_path+"train/", flowchart_path)
    _, dialogs_val, _, _, flowchart_jsons_val, docs_jsons_val = process_flowchart(dialogs_path+"val/", flowchart_path)
    _, dialogs_test, _, _, flowchart_jsons_test, docs_jsons_test = process_flowchart(dialogs_path+"test/", flowchart_path)

    combined = combine_jsons_for_tfidf(flowchart_jsons, flowchart_jsons_val, flowchart_jsons_test)
    combined_docs = combine_jsons_for_docs(docs_jsons, docs_jsons_val, docs_jsons_test)

    train_entries = create_dict(dialogs, flowchart_jsons, combined, combined_docs)
    val_entries = create_dict(dialogs_val, flowchart_jsons_val, combined, combined_docs)
    test_entries = create_dict(dialogs_test, flowchart_jsons_test, combined, combined_docs)

    scores = {'train':train_entries, 'test':test_entries, 'valid':val_entries}
    with open(save_path, 'w') as f:
        json.dump(scores,f)
    return scores

def combine_jsons_for_tfidf(flowchart_jsons, flowchart_jsons_val, flowchart_jsons_test):
    combined = deepcopy(flowchart_jsons)
    for chart, json in flowchart_jsons_val.items():
        if chart not in combined:
            combined[chart]=json
    for chart, json in flowchart_jsons_test.items():
        if chart not in combined:
            combined[chart]=json
    return combined

def combine_jsons_for_docs(docs_jsons, docs_jsons_val, docs_jsons_test):
    combined = deepcopy(docs_jsons)    
    for chart, json in docs_jsons_val.items():
        if chart not in combined:
            combined[chart]=json
    for chart, json in docs_jsons_test.items():
        if chart not in combined:
            combined[chart]=json
    return combined

def create_dict(dialogs_test, flowchart_jsons_test, combined, combined_docs):
    flows=Flowcharts(combined)
    all_chart_start = {}
    all_chart_end = {}
    all_chart_contexts = []
    all_chart_contexts_array = []
    all_chart_responses = []
    all_chart_response2node = {}
    chart_response_map = {}
    all_doc_responses = {}
    all_ranked_docs = []
    all_correct_nodes = []
    for chart in list(combined.keys()):
        chart_contexts, chart_contexts_array, chart_responses, chart_response2node, doc_response2node = get_documents_from_flowchart2(flows,chart, combined_docs[chart])
        inv_chart_response2node = {v:k+len(all_chart_responses) for k,v in chart_response2node.items()}
        all_chart_start[chart]=len(all_chart_contexts)
        all_chart_end[chart]=len(chart_contexts)+len(all_chart_contexts)
        all_chart_contexts+=chart_contexts
        all_chart_contexts_array+=chart_contexts_array
        all_chart_responses+=chart_responses
        all_chart_response2node.update(inv_chart_response2node)
        chart_response_map[chart]=chart_responses
        all_doc_responses[chart]=doc_response2node

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_chart_contexts)

    Y_tidf_map = {}
    for chart in list(flowchart_jsons_test.keys()):
        input_contexts, _, _ = get_processed_input_data(dialogs_test,chart)

        #TF-IDF
        Y_tfidf = vectorizer.transform(input_contexts)
        Y_tidf_map[chart]=Y_tfidf

    entries = {}
    for chart in list(flowchart_jsons_test.keys()):
        #chart_contexts, chart_responses, chart_response2node = get_documents_from_flowchart(flows,chart)
        input_contexts, input_responses, input_labels = get_processed_input_data(dialogs_test,chart)

        #TF-IDF
        Y_tfidf = Y_tidf_map[chart]#vectorizer.transform(input_contexts)
        if sim=='cosine':
            Y_tfidf = cosine_similarity(Y_tfidf,X[all_chart_start[chart]:all_chart_end[chart]])
        else:
            Y_tfidf = eucledian_distance(Y_tfidf,X[all_chart_start[chart]:all_chart_end[chart]])

        #BLEU
        Y_bleu = get_bleu_matrix(input_responses, chart_response_map[chart])
        #Y_tfidf = softmax(Y_tfidf,axis=-1)
        Y_tfidf = Y_tfidf/(np.max(Y_tfidf,axis=-1)[:,None]+eps)
        Y_tfidf = Y_tfidf if sim=='cosine' else 1 - Y_tfidf
        #Y_bleu = softmax(Y_bleu,axis=-1)
        Y_bleu = Y_bleu/(np.max(Y_bleu,axis=-1)[:,None]+eps)
        Y = Y_tfidf*(1-alpha)+Y_bleu*(alpha)
        #Y = Y_bleu

        ranked_docs = get_top_docs(Y,topk=1000)
        retrieved_responses = [[all_chart_responses[y+all_chart_start[chart]] for y in x] for x in ranked_docs]
        retrieved_contexts = [[all_chart_contexts_array[y+all_chart_start[chart]] for y in x] for x in ranked_docs]
        retrieved_responses_scores = [[Y[i][y] for y in x] for i,x in enumerate(ranked_docs)]

        #ranking info
        # correct_nodes = [all_chart_response2node[chart+":"+x]-all_chart_start[chart] for x in input_labels]
        all_ranked_docs += [[all_doc_responses[chart][y] if y in all_doc_responses[chart] else y for y in x] for x in ranked_docs]
        # all_correct_nodes += correct_nodes

        entries[chart] = []
        for idx in list(range(len(retrieved_responses_scores))):
            actual_response = input_responses[idx]
            context = input_contexts[idx].split(" <SEP> ")
            topk_retrieved = retrieved_responses[idx]
            topk_retrieved_context = retrieved_contexts[idx]
            topk_retrieved_scores = retrieved_responses_scores[idx]

            entry = {'Context':context,'Actual Response':actual_response, 'Retrieved Response':topk_retrieved, 'Retrieved Contexts':topk_retrieved_context, 'Scores':topk_retrieved_scores}
            entries[chart].append(entry)
        
    #calculate metrics
    # print("R@1=",np.mean([r_at_k([all_correct_nodes[i]],x,1) for i,x in enumerate(all_ranked_docs)]))
    # print("R@5=",np.mean([r_at_k([all_correct_nodes[i]],x,5) for i,x in enumerate(all_ranked_docs)]))
    return entries
    



# %%
