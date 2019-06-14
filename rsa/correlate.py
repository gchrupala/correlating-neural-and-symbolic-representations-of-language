import torch
import torch.nn as nn
import numpy as np

def rsa(A, B):
    "Returns the correlation between the similarity matrices for A and B."
    M_A = cosine_matrix(A, A)
    M_B = cosine_matrix(B, B)
    return pearson(triu(M_A), triu(M_B), dim=0) 


def cosine_matrix(U, V):
    "Returns the matrix of cosine similarity between each row of U and each row of V."
    U_norm = U / U.norm(2, dim=1, keepdim=True)
    V_norm = V / V.norm(2, dim=1, keepdim=True)
    return U_norm @ V_norm.t()


def triu(x):
    "Extracts upper triangular part of a matrix, excluding the diagonal."
    ones  = torch.ones_like(x)
    return x[torch.triu(ones, diagonal=1) == 1]


def pearson(x, y, dim=0, eps=1e-8):
    "Returns Pearson's correlation coefficient."
    x1 = x - torch.mean(x, dim)
    x2 = y - torch.mean(y, dim)
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return w12 / (w1 * w2).clamp(min=eps)
                                                                                    
def pearsonr(x, y, axis=0, eps=1e-8): 
     "Returns Pearson's correlation coefficient."
     from numpy.linalg import norm 
     x1 = x - x.mean(axis=axis) 
     x2 = y - y.mean(axis=axis) 
     w12 = np.sum(x1*x2, axis=axis) 
     w1 = norm(x1, 2, axis=axis) 
     w2 = norm(x2, 2, axis=axis) 
     return w12 / np.maximum(eps, (w1 * w2))

def abs_diff(x):
    x = x.unsqueeze(dim=0)
    return torch.abs(x - x.t())

def abs_diff_(x, y):
    x = x.unsqueeze(dim=0)
    y = y.unsqueeze(dim=0)
    return torch.abs(x.t() - y)

def shuffled(x):
    i = torch.randperm(x.size(0))
    return x[i]

class LinearRegression():
    
    def __init__(self):
        self.coef = None
        self.intercept = None
    
    def fit(X, Y):
        const = torch.ones_like(X[:, 0:1])
        X = torch.cat([const, X], dim=1)
        coef = torch.inverse(X.t() @ X) @ X.t() @ Y
        self.intercept = coef[0]
        self.coef = coef[1:]

    def predict(X):
        return X @ coef + intercept

def RSA_regress(reference, data, sim1, sim2, cv=3):
    """Embed data into similarity space and regress representation vectors according to similarity function sim2 against 
       representation vectors according to similarity function sim1."""
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr
    emb1 = sim1(data, reference)
    emb2 = sim2(data, reference)
    result = dict(pearsonr=pearsonr(emb1.reshape(-1), emb2.reshape(-1)),
                  r2_12=cross_val_score(Ridge(alpha=1.0), emb1, emb2, cv=cv),
                  r2_21=cross_val_score(Ridge(alpha=1.0), emb2, emb1, cv=cv))
    return result


def pairwise(f, data1, data2=None, normalize=False, device='cpu', dtype=torch.float32):
    """Compute matrix of values of function f applied to elements of data1 and data2."""
    symmetric = False
    if data2 is None:
        data2 = data1
        symmetric = True
    M = torch.zeros(len(data1), len(data2), dtype=dtype)
    if normalize:
        self1 = torch.tensor([f(d, d) for d in data1], dtype=dtype, device=device)
        self2 = self1 if symmetric else torch.tensor([f(d, d) for d in data2], dtype=dtype, device=device)
    for i, d1 in enumerate(data1):
        for j, d2 in enumerate(data2):
            denom = (self1[i] * self2[j])**0.5 if normalize else 1.0
            if symmetric and i > j: # No need to re-compute lower triangular
                M[i, j] = M[j, i]
            else:
                M[i, j] = f(d1, d2) / denom
    return M

def compute_value(f, self1, self2, i, j, d1, d2, normalize):
    denom = (self1[i] * self2[j])**0.5 if normalize else 1.0
    return f(d1, d2) / denom


def pairwise_parallel(f, data1, data2=None, normalize=False, device='cpu', backend="loky", n_jobs=-1, dtype=torch.float32):
    """Compute matrix of values of function f applied to elements of data1 and data2."""
    from joblib import Parallel, delayed
    symmetric = False
    if data2 is None:
        data2 = data1
        symmetric = True
    if normalize:
        self1 = torch.tensor([f(d, d) for d in data1], dtype=dtype, device=device)
        self2 = self1 if symmetric else torch.tensor([f(d, d) for d in data2], dtype=dtype, device=device)
    M = torch.tensor(Parallel(n_jobs=n_jobs, backend=backend)(delayed(compute_value)(f, self1, self2, i, j, d1, d2, normalize) 
                                    for i, d1 in enumerate(data1) for j, d2 in enumerate(data2)), dtype=dtype, device=device).view(len(data1), len(data2))
    return M

def corr_cv(model, X, Y, cv=10):
    """Compute correlation estimate for model data."""
    from sklearn.model_selection import cross_val_score
    rs = cross_val_score(model, X, Y, scoring=score_r, cv=cv, error_score=np.nan)
    return rs

def score_r(model, X, y): 
     r =  pearsonr(y, model.predict(X), axis=0).mean() 
     #r2 =  model.score(X, y)
     #print(r, r2, r2**0.5)
     return r

def score_r2(model, X, y): 
     from sklearn.metrics import r2_score
     r =  r2_score(y.reshape(-1), model.predict(X).reshape(-1))
     rmvr = r2_score(y, model.predict(X))
     print("Flat vs nested", r, rmvr)
     #r2 =  model.score(X, y)
     #print(r, r2, r2**0.5)
     return r

