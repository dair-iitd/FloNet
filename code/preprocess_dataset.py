# %%
import sys
import os
import json
import argparse

# %%
def parge_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../data/FloDial-dataset/', help='path to the dataset')
    parser.add_argument('--save_root', type=str, default='../data/formatted/', help='new path to the preprocessed dataset')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # %% [markdown]
    # ## convert flowcharts

    # %%
    args = parge_args()
    root = args.root
    save_root = args.save_root

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    flowchart_root = root + "knowledge-sources/"
    save_flowchart_root = save_root + "flowcharts/"

    if not os.path.exists(save_flowchart_root):
        os.mkdir(save_flowchart_root)

    charts = [x for x in os.listdir(flowchart_root) if '.json' in x]
    for chart in charts:
        chart = chart.replace('.json', '')
        save_path = os.path.join(save_flowchart_root,chart)
        os.makedirs(save_path,exist_ok=True)
        doc_save_path = os.path.join(save_path,"supporting_docs.json")
        save_path = os.path.join(save_path,chart+'.json')

        load_path = os.path.join(flowchart_root,chart+".json")
        with open(load_path,'r') as f:
            chart_data = json.load(f)

        doc_data = {'supporting_docs':[]}
        doc_data['supporting_faqs'] = chart_data['supporting_faqs']
        del chart_data['supporting_faqs']
        with open(save_path,"w") as f:
            json.dump(chart_data,f,indent=4)
        with open(doc_save_path,"w") as f:
            json.dump(doc_data,f,indent=4)

    # %% [markdown]
    # ## Revert json formatting and values

    # %%
    def revert_keys(data):
        for i, x in data.items():
            for j, y in enumerate(x['utterences']):
                if 'grounded_doc_id' in y and y['grounded_doc_id'] != None:
                    doc_id = y['grounded_doc_id']
                    node = None
                    if 'chart' in doc_id:
                        node = doc_id.split("-")[-1]
                    faqnode = node
                    if 'faq' in doc_id:
                        faqnode = "FAQ:"+doc_id.split("-")[-1]

                    data[i]['utterences'][j]['node'] = node
                    data[i]['utterences'][j]['FinalNode'] = faqnode
                    del data[i]['utterences'][j]['grounded_doc_id']

                if 'speaker' in y and y['speaker']=='agent':
                    data[i]['utterences'][j]['speaker'] = 'agnt'

        return data

    dialogs_root = root+"/dialogs/"
    with open(dialogs_root+ "./dialogs.json", 'r') as f:
        all_dialogs = json.load(f)
        
    dialogs = revert_keys(all_dialogs)

    with open(dialogs_root+'/formatted_dialogs.json', 'w') as f:
        json.dump(dialogs,f,indent=4)

    # %% [markdown]
    # ## convert from list to original format

    # %%
    dialogs_root = root+"/dialogs/"
    formatted_dialogs_root = save_root+"dialogs/"

    with open(dialogs_root+ "./formatted_dialogs.json", 'r') as f:
        all_dialogs = json.load(f)

    domains = {'s-flo':'in_domain_hard','u-flo':'out_domain'}
    for k_domain, v_domain in domains.items():
        with open(dialogs_root+"/"+k_domain+".json", 'r') as f:
            split_map = json.load(f)

        #make combined split
        domain_dialog_split = {}
        for k,v in split_map.items():
            save_path = os.path.join(formatted_dialogs_root,v_domain)
            os.makedirs(save_path,exist_ok=True)
            domain_dialog_split[k] = [all_dialogs[str(x)] for x in v] 
            with open(os.path.join(save_path,k+".json"),"w") as f:
                json.dump({'dialogs':domain_dialog_split[k]},f,indent=4)

        #make dataset folder
        kmap = {'trn':'train','tst':'test','val':'val'}
        for k,v in domain_dialog_split.items():
            chart_split = {}
            for x in v:
                if x['flowchart'] not in chart_split:
                    chart_split[x['flowchart']]=[]
                chart_split[x['flowchart']].append(x)
            for k1,v1 in chart_split.items():
                path = os.path.join(formatted_dialogs_root,'dataset',v_domain,kmap[k])
                os.makedirs(path,exist_ok=True)
                path = os.path.join(path, k1+'.json')

                with open(path,'w') as f:
                    json.dump({'dialogs':v1},f,indent=4)


    # %%



