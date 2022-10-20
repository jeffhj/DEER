import sys
import json
import pandas as pd
import pickle
import networkx as nx

test_dataset_file = sys.argv[1]
random_graph_file = sys.argv[2]
exp_graph_file = sys.argv[3]
sig_graph_file = sys.argv[4]
with open(random_graph_file, 'rb') as f_in:
    random_graph:nx.Graph = pickle.load(f_in)
with open(exp_graph_file, 'rb') as f_in:
    exp_graph:nx.DiGraph = pickle.load(f_in)
with open(sig_graph_file, 'rb') as f_in:
    sig_graph:nx.DiGraph = pickle.load(f_in)
    
tar = pd.DataFrame([{'pair' : 'ent1: %s ent2: %s' % tuple(item['pair']), 
                     'rdn' : random_graph.get_edge_data(*item['pair'])['data'][0]['note'], 'rdn_score' : '', 
                     'exp' : exp_graph.get_edge_data(*item['pair'])['data'][0]['note'], 'exp_score' : '', 
                     'sig' : sig_graph.get_edge_data(*item['pair'])['data'][0]['note'], 'sig_score' : '', 
                     'scr' : item['target'], 'scr_score' : ''} for item in json.load(open(test_dataset_file))])
col_names = [fname.split('.')[0] for fname in sys.argv[2:]]
gen_dfs = []
for fname in sys.argv[5:]:
    col_name = fname.split('.')[0]
    col_score = col_name + '_score'
    gen_dfs.append(pd.DataFrame({col_name:open(fname).read().splitlines()}))
    gen_dfs.append(pd.DataFrame({col_score:[''] * len(gen_dfs[-1])}))
    
# gen_dfs = [pd.DataFrame({fname.split('.')[0]:open(fname).read().splitlines()}) for fname in sys.argv[2:]]
pd.concat([tar]+gen_dfs, axis=1).to_csv('human_evaluation.csv', index=False)