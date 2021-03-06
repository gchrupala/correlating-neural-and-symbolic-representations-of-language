import random
random.seed(666)

import pylab
import numpy as np
import pandas as pd
import rsa.arithmetic as A
import rsa.experiment as E
import rsa.models as M
import rsa.kernel as K
import rsa.synsem as S
import torch.nn.functional as F
import torch
import pickle
import rsa.correlate as C
import json
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import rsa.pretrained as P
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from nltk.tree import Tree
import pandas as pd
import logging
import sys

def main(file=sys.stdout):
    # Expects experiments in experiments/er-d1.5, experiments/dr-d1.5, experiments/dec-d1.5 to have been run
    print("Figure 1 \label{fig:sim_distribution}\n", file=file)
    sim_distribution()
    subprocess.run(["R", "--vanilla", "--slave", "-e", "source('plot.R'); sim_distribution()"])
    print("\includegraphics[scale=0.5]{sim_distribution.png}\n", file=file)

    print("Figure 5 \label{fig:size_dist}\n", file=file)
    size_distribution()
    subprocess.run(["R", "--vanilla", "--slave", "-e", "source('plot.R'); size_distribution()"])
    print("\includegraphics[scale=0.35]{size_distribution.png}\n", file=file)

    print("Figure 6 \label{fig:scatter_rsa_prefix}\n", file=file)
    try:
        scorer = pickle.load(open("report/scorer.pkl", "rb"))
    except FileNotFoundError:
        data = pickle.load(open("report/data_test_ref.pkl", "rb"))
        #data = make_test_data(decay=1.5, size=2000, directory="report")
        #data = make_test_data(decay=1.5, size=100, directory="report")
        scorer = make_scorer(data['test'], data['ref']) 
        pickle.dump(scorer, open("report/scorer.pkl", "wb"))
    scatter_rsa()
    subprocess.run(["R", "--vanilla", "--slave", "-e", "source('plot.R'); scatter_rsa()"])
    print("\includegraphics[scale=0.3]{scatter_rsa.png}\n", file=file)
    
    print("Table 2 \label{tab:results_main}\n", file=file)
    tables = synthetic_results(scorer)
    for name, table in tables.items():
        print(name +"\n", file=file)
        print(table.to_latex(), file=file)


    print("Table 3 \label{tab:rsa-natural}\n", file=file)
    RSA_natural()
    natural_TK_summary(file=file)

    print("Figure 7 \label{fig:bert-layers}\n", file=file)
    BERT_layers()
    subprocess.run(["R", "--vanilla", "--slave", "-e", "source('plot.R'); BERT_layers(24)"])
    print("\includegraphics[scale=0.45]{BERT24_layers.png}\n", file=file)
    
def synthetic_results(scorer):
    out = {}
    conditions = [("Semantic evaluation", "experiments/er-d1.5"), ("Tree Depth", "experiments/dr-d1.5"), ("Infix-to-Prefix", "experiments/dec-d1.5")]
    for name, d in conditions:
        out[name] = report_synth(name, d, scorer)
    json.dump(make_json_happy(out), open("report/synthetic.json", "w"), indent=2)
    return tabulate(out)


def tabulate(r):
    encoders = ["Random", "Semantic evaluation", "Tree Depth", "Infix-to-Prefix"]
    row = dict([('Random', r['Semantic evaluation']['init'])] + [(encoder, r[encoder]['best']) for encoder in encoders[1:]])
    diag = pd.DataFrame(
        dict(Encoder=encoders,
             Loss   = [ row[encoder].get('loss', np.nan) for encoder in encoders ],
             Value  = [ row[encoder]['diagnostic']['r_value']['mean'] for encoder in encoders ],
             Depth  = [ row[encoder]['diagnostic']['r_depth']['mean'] for encoder in encoders ]))
    rsa = pd.DataFrame(
        { 'Encoder': encoders,
          'Value':   [ row[encoder]['rsa'][1.0]['rep_value'] for encoder in encoders ],
          'Depth':   [ row[encoder]['rsa'][1.0]['rep_depth'] for encoder in encoders ],
          'TK(1.0)': [ row[encoder]['rsa'][1.0]['rep_tree'] for encoder in encoders ],
          'TK(0.5)': [ row[encoder]['rsa'][0.5]['rep_tree'] for encoder in encoders ] })
    reg = pd.DataFrame(
        { 'Encoder': encoders,
          'Value':   [ row[encoder]['rsa_regress'][1.0]['value']['r_12']['mean']  for encoder in encoders ],
          'Depth':   [ row[encoder]['rsa_regress'][1.0]['depth']['r_12']['mean']  for encoder in encoders ],
          'TK(1.0)': [ row[encoder]['rsa_regress'][1.0]['tree']['r_12']['mean']   for encoder in encoders ],
          'TK(0.5)': [ row[encoder]['rsa_regress'][0.5]['tree']['r_12']['mean']   for encoder in encoders ] })
    return { 'Diagnostic': diag, 'RSA': rsa, 'RSAregress': reg }
             
    
def report_synth(name, directory, scorer):
    path0 = '{}/net-d2.0-00.pkl'.format(directory)
    net0 = pickle.load(open(path0, 'rb'))
    path, net = best_weights(directory, data=scorer.data)
    result = { 'name': name,
               'init': scorer.score(lambda E: net0.encode(E)),
               'best': scorer.score(lambda E: net.encode(E)) }
    result['init']['path'] = path0
    result['best']['path'] = path
    result['best']['loss'] = score(net, scorer.data).item()
    return result

def RSA_natural():
  try:
      data_sent = json.load(open("data/out/ewt.json"))
  except FileNotFoundError:
      S.ewt_json()
      data_sent = json.load(open("data/out/ewt.json"))
  try:
      data = torch.load("data/out/ewt_embed.pt")
  except FileNotFoundError:
      S.ewt_embed()
      data = torch.load("data/out/ewt_embed.pt")

  result = {}

  for alpha in [0.5, 0.75, 1.0]:
    try:
        data_tk = torch.load("data/out/ewt-pairwise-TK-alpha-{}.pt".format(alpha))
    except FileNotFoundError:
        #S.ewt_TK_old(parallel=True, alpha=alpha)
        S.ewt_TK(alpha=alpha)
        data_tk = torch.load("data/out/ewt-pairwise-TK-alpha-{}.pt".format(alpha))
    data_tk['test_test'] = data_tk['test_test'].float()
    data_tk['test_ref'] = data_tk['test_ref'].float()


    result[alpha] = dict(bow=dict(), bert=dict(), bert24=dict(), infersent=dict())
    
    data_enc_bow = dict(test=data['bow']['test'], ref=data['bow']['ref'])
    result[alpha]['bow']=RSA_report(data_tk, data_enc_bow)
    result[alpha]['bert']=dict(random={}, trained={})
    result[alpha]['bert24']=dict(random={}, trained={})
    result[alpha]['infersent'] = dict(random={}, trained={})

    for mode in ['random', 'trained']:
        for step in ['first', 'last']:
            result[alpha]['bert'][mode][step] = {}
            result[alpha]['bert24'][mode][step] = {}
            for layer in range(12):
                logging.info("Computing RSA/RSA_regress scores for {} {} {}".format(mode, step, layer))
                data_enc = dict(test=data['bert']['test'][mode][layer][step], ref=data['bert']['ref'][mode][layer][step])               
                result[alpha]['bert'][mode][step][layer] = RSA_report(data_tk, data_enc)
            for layer in range(24):
                logging.info("Computing RSA/RSA_regress scores for {} {} {}".format(mode, step, layer))
                data_enc = dict(test=data['bert24']['test'][mode][layer][step], ref=data['bert24']['ref'][mode][layer][step])               
                result[alpha]['bert24'][mode][step][layer] = RSA_report(data_tk, data_enc)

        result[alpha]['infersent'][mode] = RSA_report(data_tk, dict(test=data['infersent']['test'][mode], ref=data['infersent']['ref'][mode]))
  json.dump(result, open("report/RSA_natural.json", "w"), indent=2)


def natural_TK_summary(file):
    import rsa.pretrained as P
    train = dict(random="$-$", trained="$+$")
    data = json.load(open("report/RSA_natural.json")) 
    rows = []
    for alpha in ['0.5', '1.0']: 
        # BOW    
        rows.append(["BoW      ", "      ", alpha, data[alpha]['bow']['rsa'], data[alpha]['bow']['rsa_regress']['pearson_r']['mean']])
        for mode in ['random', 'trained']:
            rows.append(["Infersent", train[mode], alpha, data[alpha]['infersent'][mode]['rsa'], 
                        data[alpha]['infersent'][mode]['rsa_regress']['pearson_r']['mean']])
            best_reg = max(range(24), key=lambda i: data[alpha]['bert24'][mode]['first'][str(i)]['rsa_regress']['pearson_r']['mean'])
            best_rsa = max(range(24), key=lambda i: data[alpha]['bert24'][mode]['first'][str(i)]['rsa'])
            rows.append(["BERT last", train[mode], alpha, data[alpha]['bert24'][mode]['first']['23']['rsa'], 
                        data[alpha]['bert24'][mode]['first']['23']['rsa_regress']['pearson_r']['mean']])
            rows.append(["BERT best", train[mode], alpha, data[alpha]['bert24'][mode]['first'][str(best_rsa)]['rsa'], 
                        data[alpha]['bert24'][mode]['first'][str(best_reg)]['rsa_regress']['pearson_r']['mean']])

    print(r"\begin{tabular}{llrrr}"+"\n", file=file)
    for row in rows:
            print("  &  ".join(row[:3] + ["{:.2f}".format(x) for x in row[3:]]) + "  \\\\", file=file)
    print("\end{tabular}\n", file=file)
    
def BERT_layers():
    with open("report/BERT_layers.csv", 'w') as f:
        data = json.load(open("report/RSA_natural.json")) 
        print("alpha encoder mode step layer metric r", file=f)
        for alpha in ['0.5', '0.75', '1.0']: 
       
          for mode in ["random", "trained"]:
            for i in range(12):
                for step in ["first", "last"]:
                    print("{} BERT {} {} {} RSA {}".format(alpha, mode, step, i+1, data[alpha]['bert'][mode][step][str(i)]['rsa']), file=f)
                    print("{} BERT {} {} {} RSA_regress {}".format(alpha, mode, step, i+1, 
                        data[alpha]['bert'][mode][step][str(i)]['rsa_regress']['pearson_r']['mean']), file=f)

            for i in range(24):
                for step in ["first", "last"]:
                    print("{} BERT24 {} {} {} RSA {}".format(alpha, mode, step, i+1, data[alpha]['bert24'][mode][step][str(i)]['rsa']), file=f)
                    print("{} BERT24 {} {} {} RSA_regress {}".format(alpha, mode, step, i+1, 
                        data[alpha]['bert24'][mode][step][str(i)]['rsa_regress']['pearson_r']['mean']), file=f)


from rsa.correlate import corr_cv

from scipy.stats import sem

def cor(a, b):
        return C.pearson(C.triu(a), C.triu(b), dim=0).item()

def rsa_regress(emb1, emb2, cv=10):
        a, r2 = tune(emb1, emb2, cv=cv)
        return dict(r_rsa=pearsonr(emb1.reshape(-1), emb2.reshape(-1)),
                    r_regress=dict(alpha=a, summary=r2))
def summary(x):
        return dict(mean=x.mean(), std=x.std(ddof=1), se=sem(x, ddof=1))

def criterion(z):
        a, x = z
        c = x.mean()
        return c        

def tune(emb1, emb2, cv=10, alpha=[ 10**n for n in range(-3,2) ]):
        r2 = ((a, corr_cv(Ridge(alpha=a), emb1, emb2, cv=cv)) for a in alpha )
        a, best = sorted(r2, key=criterion, reverse=True)[0]
        return a, summary(best) 

def depth_report(data, target, alpha, cv=10):
    print(target)
    a, r2 = tune(data, target, cv=cv, alpha=alpha)
    return dict(alpha=a, summary=r2)
    
def RSA_report(data_tk, data_enc, cv=10):
    """Compute RSA and RSA_regress scores against precomputed target symbolic similarity matrix."""
    from ursa.regress import Regress

    # RSA
    D_rep = 1-C.cosine_matrix(data_enc['test'], data_enc['test'])
    D_tk = 1 -  data_tk['test_test'].float()
    rsa = cor(D_rep, D_tk)
    # RSA_regress
    D_rep = 1-C.cosine_matrix(data_enc['test'], data_enc['ref']).detach().cpu().numpy()
    D_tk  = 1 - data_tk['test_ref'].float().detach().cpu().numpy()
    r = Regress(cv=cv)
    rsa_reg = r.fit_report(D_rep, D_tk)
    return make_json_happy(dict(rsa=rsa, rsa_regress=rsa_reg))

def sim_distribution():
    with open("report/sim_distribution.txt","w") as f:
        print("α K", file=f)
        for alpha in [0.5, 1.0]:
            try:
                data = random.sample(list(C.triu(torch.load("data/out/ewt-pairwise-TK-alpha-{}.pt".format(alpha))['test_test'].float()).cpu().numpy()), 10000)
            except FileNotFoundError:
                S.ewt_TK(alpha=alpha)
                data = random.sample(list(C.triu(torch.load("data/out/ewt-pairwise-TK-alpha-{}.pt".format(alpha))['test_test'].float()).cpu().numpy()), 10000)
            for datum in data:
                print("{} {}".format(alpha, datum), file=f)
                    

def make_test_data(decay=1.5, size=2000, directory="."):
    # test data
    _, data_test = A.generate_batch(p=1.0, decay=decay, size=size)
    # reference data
    _, data_ref = A.generate_batch(p=1.0, decay=decay, size=size//10)
    pickle.dump(data_test, open(directory + "/data_test.pkl", "wb"))
    pickle.dump(data_ref, open(directory + "/data_ref.pkl", "wb"))
    data = {'test': data_test, 'ref': data_ref}
    pickle.dump(data, open(directory + "/data_test_ref.pkl", "wb"))
    return data

def size_distribution():
    def size(d):
        return np.array([A.generate(p=1.0, decay=d).size() for _ in range(10000)])
    data = pd.DataFrame({'d_2.0': size(2.0),
                         'd_1.8': size(1.8),
                         'd_1.5': size(1.5)})
    data.to_csv('report/size_distribution.csv', index=False)

def scatter_rsa(alpha=0.5):
    scorer = pickle.load(open("report/scorer.pkl","rb"))
    path0 = 'experiments/dec-d1.5/net-d2.0-00.pkl'
    net0 = pickle.load(open(path0, 'rb'))
    _, net1 = best_weights('experiments/dec-d1.5/', data=scorer.data)
    rep0 = net0.encode(scorer.data)
    rep1 = net1.encode(scorer.data)     
    D_rep0 = C.triu(1-C.cosine_matrix(rep0, rep0)).detach().cpu().numpy()
    D_rep1 = C.triu(1-C.cosine_matrix(rep1, rep1)).detach().cpu().numpy()
    D_tree = C.triu(scorer.D_tree[alpha]).detach().cpu().numpy()
    pd.DataFrame(dict(rep0=D_rep0, rep1=D_rep1, tree=D_tree)).to_csv("report/scatter_rsa.csv", header=True, index=False)  
    
def make_scorer(data, reference):
    return E.Scorer(data,  reference=reference,
                            depth_f=lambda e: e.depth(), 
                             value_f=lambda e: e.evaluate(10), 
                             kernel_f=lambda T1, T2=None, **kwargs : K.pairwise(T1, T2=T2, normalize=True, ignore_leaves=False, **kwargs),
                             alphas = (1.0, 0.5),
                             device='cuda')


def best_weights(directory, data=None):
    if data is None:
        data = pickle.load(open('{}/data.pkl'.format(directory), 'rb'))
    # choose best model according to its own loss
    names = []
    nets = []
    for decay in [2.0, 1.8, 1.5]:
        for i in range(1, 41):
            name = '{}/net-d{}-{:02d}.pkl'.format(directory, decay, i)
            net = loadit(name)
            if net is not None:
                names.append(name)
                nets.append(net)
    
    loss = np.array([ score(net, data).item() for net in nets ]) 
    min_i = loss.argmin()
    return names[min_i], nets[min_i]
    

def loadit(x): 
     try: 
         net = pickle.load(open(x, 'rb')) 
         net.encoder.RNN.flatten_parameters()
         return net
     except FileNotFoundError: 
         return None 
          

def score(net, data):
    with torch.no_grad():
        return net.loss(data)

def depth(data):
    return torch.tensor([e.depth() for e in data], device='cuda', dtype=torch.float) 

def value(data):
    return torch.tensor([e.evaluate(10) for e in data], device='cuda', dtype=torch.float) 

def make_json_happy(x):
    if isinstance(x, np.float32) or isinstance(x, np.float64):
        return float(x)
    elif isinstance(x, dict):
        return {key: make_json_happy(value) for key, value in x.items() }
    elif isinstance(x, list):
        return [ make_json_happy(value) for value in x ]
    elif isinstance(x, tuple):
        return tuple(make_json_happy(value) for value in x)
    else:
        return x

