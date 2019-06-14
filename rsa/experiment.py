import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import rsa.arithmetic as A
import rsa.correlate as C
import rsa.models as M
import rsa.kernel as K
from collections import Counter
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from rsa.correlate import corr_cv

def accuracy(tgt, pred):
    return torch.mean((tgt == pred).float()).item()


def score(U, V):
    u = C.triu(U).double()
    v = C.triu(V).double()
    return C.pearson(u, v, dim=0).item()

def summary(x):
    return dict(mean=x.mean(), std=x.std())

class Scorer:
    
    def __init__(self, data, reference=None,
                             depth_f=lambda e: e.depth(), 
                             value_f=lambda e: e.evaluate(10), 
                             kernel_f=lambda E, **kwargs: K.pairwise(E, normalize=True, ignore_leaves=True, **kwargs),
                             alphas = (1.0, 0.5),
                             device='cuda'):
        self.data = data
        self.alphas = alphas
        if reference is not None:
            self.reference = reference
        else:
            _, self.reference = A.generate_batch(p=1.0, decay=1.5, size=200)
        self.depth = torch.tensor([depth_f(e) for e in data], dtype=torch.float32, device=device)
        self.value = torch.tensor([value_f(e) for e in data], dtype=torch.float32, device=device)
        self.D_tree = { alpha: 1-kernel_f(self.data, alpha=alpha).to(device=device) for alpha in self.alphas }
        self.D_depth = C.abs_diff(self.depth)
        self.D_value  = C.abs_diff(self.value)
        self.tree_depth = { alpha: score(self.D_tree[alpha], self.D_depth) for alpha in self.alphas }
        self.tree_value  = { alpha: score(self.D_tree[alpha], self.D_value) for alpha in self.alphas }
        self.depth_value = score(self.D_depth, self.D_value)
        self.emb_tree  = { alpha: 1-kernel_f(self.data, self.reference, alpha=alpha).detach().cpu().numpy() for alpha in self.alphas }
        self.emb_depth = C.abs_diff_(self.depth, torch.tensor([depth_f(e) for e in self.reference], dtype=torch.float32, device=device)).detach().cpu().numpy()
        self.emb_value = C.abs_diff_(self.value, torch.tensor([value_f(e) for e in self.reference], dtype=torch.float32, device=device)).detach().cpu().numpy()

    def score(self, encode_f):
        rep = encode_f(self.data)
        D_rep = 1-C.cosine_matrix(rep, rep)
        emb = 1-C.cosine_matrix(rep, encode_f(self.reference)).detach().cpu().numpy()
        X_rep = rep.detach().cpu().numpy()
        y_value = self.value.detach().cpu().numpy()
        y_depth = self.depth.detach().cpu().numpy()
        r_value  = corr_cv(Ridge(alpha=1.0), X_rep, y_value, cv=10)
        r_depth  = corr_cv(Ridge(alpha=1.0), X_rep, y_depth, cv=10)
        acc_value = cross_val_score(LogisticRegression(C=1.0, solver='liblinear', multi_class='auto'), X_rep, y_value, cv=10)
        acc_depth = cross_val_score(LogisticRegression(C=1.0, solver='liblinear', multi_class='auto'), X_rep, y_depth, cv=10)
        return dict(rsa={ alpha: dict(rep_tree =  score(D_rep, self.D_tree[alpha]),
                                      rep_depth=   score(D_rep, self.D_depth),
                                      rep_value =  score(D_rep, self.D_value),
                                      tree_depth = self.tree_depth[alpha],
                                      tree_value = self.tree_value[alpha],
                                      depth_value = self.depth_value) for alpha in self.alphas },
                    diagnostic=dict(r_depth=summary(r_depth), 
                                    r_value=summary(r_value), 
                                    acc_depth=summary(acc_depth), 
                                    acc_value=summary(acc_value)),
                    rsa_regress={ alpha: dict(tree=rsa_regress(emb, self.emb_tree[alpha], cv=10),
                                              value= rsa_regress(emb, self.emb_value, cv=10),
                                              depth=rsa_regress(emb, self.emb_depth, cv=10)) for alpha in self.alphas } #### FIXME FIXME
                    )

def rsa_regress(emb1, emb2, cv=10):
    return dict(r_12=summary(corr_cv(Ridge(alpha=1.0), emb1, emb2, cv=cv)),
                r_21=summary(corr_cv(Ridge(alpha=1.0), emb2, emb1, cv=cv)))

def get_device(net):
    return list(net.parameters())[0].device



