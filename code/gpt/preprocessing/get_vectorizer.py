import json
import pickle
from .Flowcharts import Flowcharts
from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir
import os
from copy import deepcopy

PATH_SEPARATOR = " <SEP> "

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

def get_documents_from_flowchart1(flows,chart):
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
        chart_response2node[len(chart_responses)-1]=chart+":"+node
    return chart_contexts, chart_responses, chart_response2node

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
        doc_response2node[len(chart_responses)-1]=inv_chart_response2node[chart+":"+d['node']]
    return chart_contexts, chart_contexts_array, chart_responses, chart_response2node, doc_response2node


def save_vectorizer(root,domain,save_path,chart_path='../data/flowcharts/'):
    
    _, dialogs, _, _, flowchart_jsons, docs_jsons = process_flowchart(root+domain+"/train/", chart_path)
    _, dialogs_val, _, _, flowchart_jsons_val, docs_jsons_val = process_flowchart(root+domain+"/val/", chart_path)
    _, dialogs_test, _, _, flowchart_jsons_test, docs_jsons_test = process_flowchart(root+domain+"/test/", chart_path)

    combined = combine_jsons_for_tfidf(flowchart_jsons, flowchart_jsons_val, flowchart_jsons_test)
    combined_docs = combine_jsons_for_docs(docs_jsons, docs_jsons_val, docs_jsons_test)

    flows=Flowcharts(combined)

    chart_context_dict = {}
    chart_response_dict = {}
    all_chart_contexts = []
    for chart in list(flowchart_jsons_test.keys()):
        chart_contexts, _, chart_responses, _, _ = get_documents_from_flowchart2(flows,chart, combined_docs[chart])
        all_chart_contexts+=chart_contexts
        chart_response_dict[chart]=chart_responses
        chart_context_dict[chart]=chart_contexts

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_chart_contexts)
    tfidf_data = {"vectorizer":vectorizer,
                    "contexts":chart_context_dict,
                    "responses":chart_response_dict}
    with open(save_path, 'wb') as f:
        pickle.dump(tfidf_data,f)

if __name__=="__main__":
    save_path="chart_vectorizer.pkl"
    domain="in_domain_hard"
    root = "./data/flodial/dataset/"
    chart_path = "./data/flodial/dataset/"
    save_vectorizer(root,domain,save_path)
