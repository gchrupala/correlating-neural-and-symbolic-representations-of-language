from xml.etree import ElementTree as ET
import pickle
import subprocess
import json
import random
import torch
import rsa.correlate as C
import sys
import pandas as pd
from nltk.tree import Tree
import conllu as U


def ewt_json():
    def id2path(sentid, prefix="data/in/LDC2012T13/eng_web_tbk/data/"):
        cols = sentid.split('-')
        return (prefix + cols[0] + "/penntree/" + '-'.join(cols[1:-1]) + ".xml.tree", int(cols[-1])-1)
    def get_tree(sentid):
        path, index = id2path(sentid)
        return [Tree.fromstring(line) for line in open(path) ][index]
    test =  U.parse(open("data/in/UD_English-EWT-master/en_ewt-ud-dev.conllu").read())
    train = U.parse(open("data/in/UD_English-EWT-master/en_ewt-ud-train.conllu").read())
    ref = random.sample(train, len(test)//10)
    data_test = [ dict(sent=datum.metadata['text'], sentid=datum.metadata['sent_id'], tree=str(get_tree(datum.metadata['sent_id']))) for datum in test ]
    data_ref  = [ dict(sent=datum.metadata['text'], sentid=datum.metadata['sent_id'], tree=str(get_tree(datum.metadata['sent_id']))) for datum in ref ]    
    json.dump(dict(ref=data_ref, test=data_test), open("data/out/ewt.json","w"))

def ewt_TK(alpha=1.0):
    """Compute pairwise tree kernel scores for EWT data and save to file."""
    import numpy as np
    from ursa.kernel import Kernel, delex
    try:
        data = json.load(open("data/out/ewt.json"))
    except FileNotFoundError:
        ewt_json()
        data = json.load(open("data/out/ewt.json"))
    ref = [ delex(Tree.fromstring(datum['tree'])) for datum in data['ref'] ]
    test = [ delex(Tree.fromstring(datum['tree'])) for datum in data['test'] ]        
    M_test_test = Kernel(alpha=alpha).pairwise(test, normalize=True, dtype=np.float64)
    M_test_ref  = Kernel(alpha=alpha).pairwise(test, ref, normalize=True, dtype=np.float64)
    torch.save(dict(test_test=torch.tensor(M_test_test), test_ref=torch.tensor(M_test_ref)),  "data/out/ewt-pairwise-TK-alpha-{}.pt".format(alpha))

def ewt_embed():
    """Compute BoW, BERT and Infersent embeddings for the EWT data and save to file."""
    import rsa.pretrained as Pre
    from sklearn.feature_extraction.text import CountVectorizer
    def container():
        return dict(test=dict(random=dict(), trained=dict()), 
                    ref=dict(random=dict(), trained=dict()))
    data = json.load(open("data/out/ewt.json"))
    emb = dict(bow={}, bert=container(), bert24=container(), infersent=container())
    # BOW 
    v = CountVectorizer(tokenizer=lambda x: x.split())
    sent_ref = [s['sent'] for s in data['ref'] ]
    sent_test = [s['sent'] for s in data['test'] ]
    v.fit( sent_ref + sent_test ) 
    emb['bow']['test'] = torch.tensor(v.transform(sent_test).toarray(), dtype=torch.float)
    emb['bow']['ref'] =  torch.tensor(v.transform(sent_ref).toarray(), dtype=torch.float)

    for split in ['test', 'ref']:
      sent = [ datum['sent'] for datum in data[split] ]
      for mode in ["random", "trained"]:
        if mode == "random":
            rep24 = list(Pre.encode_bert(sent, trained=False, large=True))
            rep = list(Pre.encode_bert(sent, trained=False))
            emb['infersent'][split][mode] = Pre.encode_infersent(sent, trained=False)
        else:
            rep24 = list(Pre.encode_bert(sent, trained=True, large=True))
            rep = list(Pre.encode_bert(sent, trained=True))
            emb['infersent'][split][mode] =  Pre.encode_infersent(sent, trained=True)
            
        pooled24 = torch.cat([ pooled for _, pooled in rep24 ])
        pooled = torch.cat([ pooled for _, pooled in rep ])
        emb['bert24'][split][mode]['pooled'] = pooled24
        emb['bert'][split][mode]['pooled'] = pooled
        for i in range(len(rep24[0][0])):
            emb['bert24'][split][mode][i] = {}
            emb['bert24'][split][mode][i]['summed'] = torch.cat([ layers[i].sum(dim=1) for layers, _ in rep24 ], dim=0)
            emb['bert24'][split][mode][i]['first']  = torch.cat([ layers[i][:,0,:] for layers, _ in rep24], dim=0)
            emb['bert24'][split][mode][i]['last']   = torch.cat([ layers[i][:,-1,:] for layers, _ in rep24], dim=0)

        for i in range(len(rep[0][0])):
            emb['bert'][split][mode][i] = {}
            emb['bert'][split][mode][i]['summed'] = torch.cat([ layers[i].sum(dim=1) for layers, _ in rep ], dim=0)
            emb['bert'][split][mode][i]['first']  = torch.cat([ layers[i][:,0,:] for layers, _ in rep], dim=0)
            emb['bert'][split][mode][i]['last']   = torch.cat([ layers[i][:,-1,:] for layers, _ in rep], dim=0)
    torch.save(emb, "data/out/ewt_embed.pt")


