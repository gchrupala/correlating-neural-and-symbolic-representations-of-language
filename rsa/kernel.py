from functools import reduce
import rsa.arithmetic as A
import logging

def match(n1, n2):
    return (n1.is_leaf() and n2.is_leaf() and n1.value == n2.value) \
                or \
           (not n1.is_leaf() and not n2.is_leaf() and n1.op == n2.op)


def C(n1, n2, alpha=1):
    # both leaves
    if n1.is_leaf() and n2.is_leaf() and match(n1, n2):
        return alpha
    # productions same and internal
    elif match(n1, n2) and match(n1.left, n2.left) and match(n1.right, n2.right):
        return alpha * (1 + C(n1.left, n2.left, alpha=alpha)) * (1 + C(n1.right, n2.right, alpha=alpha))
    else:
        return 0
            

def K(T1, T2, alpha=1.0):
    N = sum(C(n1, n2, alpha=alpha) for n1 in T1.nodes() for n2 in T2.nodes())
    return N

def deleaf(t):
    if t.is_leaf():
        return A.Expression(None)
    else:
        return A.Expression(t.op, left=deleaf(t.left), right=deleaf(t.right))

def pairwise_old(T, normalize=False, ignore_leaves=False, device='cpu', alpha=1.0):
    import torch
    if ignore_leaves:
        T = [ deleaf(t) for t in T ]
    M = torch.zeros(len(T), len(T), dtype=torch.float32, device=device)
    for i in range(len(T)):
        for j in range(len(T)):
            if i <= j:
                M[i,j] = K(T[i], T[j], alpha=alpha)
    if normalize:
        diag = M.diag().unsqueeze(dim=0)
        denom = (diag * diag.t())**0.5
        return M/denom
    else:
        return M

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

# def etree_(e):
#     "What we claim in the paper."
#     from nltk.tree import Tree
#     if e.is_leaf(): 
#          return Tree('E', [str(e.value)]) 
#     else: 
#          return Tree('E', ['(', etree(e.left), e.op, etree(e.right), ')'])
# def etree_(e):
#     "What we claim in the paper with a twist."
#     from nltk.tree import Tree
#     if e.is_leaf(): 
#          return Tree('E', [str(e.value)]) 
#     else: 
#          return Tree(e.op, ['(', etree(e.left), e.op, etree(e.right), ')'])
def _etree(e):
    "This one is the obvious rep."
    from nltk.tree import Tree
    if e.is_leaf(): 
         return Tree('D', [str(e.value)]) 
    else: 
         return Tree('E', [Tree('L', ['(']), etree(e.left), Tree('O', [e.op]), etree(e.right), Tree('R', [')'])])

def etree(e):
    "This one is the obvious rep."
    from nltk.tree import Tree
    if e.is_leaf(): 
         return Tree('E', [Tree('D', [str(e.value)])])
    else: 
         return Tree('E', [Tree('L', ['(']), etree(e.left), Tree('O', [e.op]), etree(e.right), Tree('R', [')'])])
    
# def etree_(e):
#     "This one is very close to original results"
#     from nltk.tree import Tree 
#     if e.is_leaf():  
#         return Tree('D', [str(e.value)]) 
#     else:  
#         return Tree(e.op, [etree(e.left), etree(e.right)])
    
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

