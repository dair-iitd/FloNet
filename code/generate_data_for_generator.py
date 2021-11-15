##only to be used for following models
# - Only dialog History = type='flowchart'
# - TFIDF model = type = 'dialog'
###

from gpt.preprocessing.get_vectorizer import save_vectorizer
from gpt.preprocessing.preprocess import create_gpt_input

root = "../data/new_dialogs/dataset/"
domain = "in_domain_hard"
save_path = "../data/saved_data/chart_vectorizer.pkl"
save_vectorizer(root, domain, save_path) #uncomment this to create the vectorizer pickle file

tf_idf_path=save_path
root_path=root
domain = "out_domain"
graph_path='../data/new_dialogs/flowcharts/'
save_root = "../data/saved_data/"
type = "dialog"#'tfidf' or 'dialog'
create_gpt_input(root_path,domain,tf_idf_path,graph_path,save_root,type=type)
#create_gpt_input(root_path,domain,tf_idf_path,graph_path,save_root,type='flowchart',dummy_data=True)#when you want to use a small set for testing