from functools import reduce
import rsa.arithmetic as A
import logging


def deleaf(t):
    if t.is_leaf():
        return A.Expression(None)
    else:
        return A.Expression(t.op, left=deleaf(t.left), right=deleaf(t.right))

def pairwise(T1, T2=None, normalize=False, ignore_leaves=False, device='cpu', alpha=1.0):
    import torch
    import numpy as np
    from ursa.kernel import Kernel
    if ignore_leaves:
        T1 = [ deleaf(t) for t in T1 ]
        if T2 is not None:
            T2 = [ deleaf(t) for t in T2 ]
    T1 = [ etree(t) for t in T1 ]
    if T2 is not None:
        T2 = [ etree(t) for t in T2 ]
    K = Kernel(alpha=alpha)
    return torch.tensor(K.pairwise(T1, T2, normalize=normalize, dtype=np.float64))

def etree(e):
    "This one is the obvious rep."
    from nltk.tree import Tree
    if e.is_leaf(): 
         return Tree('E', [Tree('D', [str(e.value)])])
    else: 
         return Tree('E', [Tree('L', ['(']), etree(e.left), Tree('O', [e.op]), etree(e.right), Tree('R', [')'])])
    
def cosine(A): 
     M = A @ A.t() 
     diag = torch.diag(M).unsqueeze(dim=0) 
     denom = diag**0.5 * diag.t()**0.5 
     return M/denom 

def node_overlap(a, b): 
     s = 0 
     for n1 in a.nodes(): 
         for n2 in b.nodes(): 
             if n1 == n2: 
                 s += 1 
     return s 

