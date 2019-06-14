import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import rsa.arithmetic as A
class Encoder(nn.Module):

    def __init__(self, size_in=15, size=64, depth=1, size_embed=32):
        super(Encoder, self).__init__()
        self.size_in = size_in
        self.size = size
        self.depth = depth
        self.size_embed = size_embed
        self.Embed  = nn.Embedding(self.size_in, self.size_embed) 
        self.RNN   = nn.LSTM(self.size_embed, self.size, self.depth, batch_first=True)


    def forward(self, text):
        out, last  = self.RNN(self.Embed(text))
        h_last, c_last = last
        return h_last

    
    
class Decoder(nn.Module):
    """Decoder conditioned on source and on target at t-1."""

    def __init__(self, size_in=15, size=64, depth=1, size_embed=32):
        super(Decoder, self).__init__()
        self.size_in = size_in
        self.size = size
        self.depth = depth
        self.size_embed = size_embed
        self.Embed  = nn.Embedding(self.size_in, self.size_embed) 
        self.RNN = nn.LSTM(self.size_embed, self.size, self.depth, batch_first=True)
        self.Proj = nn.Linear(self.size, self.size_in)
        
    def forward(self, rep, prev):
        c_0 = torch.zeros_like(rep)
        out, last = self.RNN(self.Embed(prev), (rep, c_0))
        pred = self.Proj(out)
        return pred

class MLP(nn.Module):
    """2-layer multi layer perceptron."""
    
    def __init__(self, size_in, size, size_out):
        super(MLP, self).__init__()
        self.L1 = nn.Linear(size_in, size)
        self.L2 = nn.Linear(size, size_out)

    def forward(self, x):
        return self.L2(torch.tanh(self.L1(x))) 


class Net(nn.Module):

    def __init__(self, encoder, n_classes=None, target_type='value', lr=0.001):
        super(Net, self).__init__()
        self.n_classes = n_classes
        self.target_type = target_type
        self.encoder = encoder
        if n_classes is not None:
            self.MLP =  MLP(self.encoder.size, self.encoder.size*2, self.n_classes)
        else:
            self.MLP = MLP(self.encoder.size, self.encoder.size*2, 1)
    
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        rep = self.encoder(x)
        pred = self.MLP(rep)
        return pred.squeeze()

    def target(self, e):
        if self.target_type == 'value':
            return e.evaluate(10)
        elif self.target_type == 'depth':
            return e.depth()
        else:
            raise NotImplementedError()

    def encode(self, expressions):
        return encode(self, expressions)
    
    def loss(self, es, length=None):
        if length is None:
            length = max([ len(list(e.infix())) for e in es ])
        loss_f=F.cross_entropy if self.n_classes is not None else F.mse_loss
        target_dtype=torch.long if self.n_classes is not None else torch.float32
        data_in  = torch.tensor([ list(A.vocabify(e.infix(), length=length)) for e in es ], dtype=torch.long, device=get_device(self)) 
        tgt = torch.tensor([self.target(e) for e in es ], dtype=target_dtype, device=get_device(self))
        pred = self(data_in).squeeze()
        loss = loss_f(pred, tgt)
        return loss
     
    def experiment(net, batch_size=16, n=1000, interval=100, decay=1.5, prefix=''):
        loss_f=F.cross_entropy if net.n_classes is not None else F.mse_loss
        cost = Counter(dict(cost=0,N=0))
        for i in range(n):
            L, es = A.generate_batch(p=1.0, decay=decay, size=batch_size)
            loss = net.loss(es, length=L)
            net.opt.zero_grad()
            loss.backward()
            net.opt.step()
            cost += Counter(dict(cost=loss.data.item(), N=1))
            if i % interval == 0:
                print(prefix, i, loss.data.item(), cost['cost']/cost['N'])

class Net_struct(nn.Module):
    def __init__(self, encoder, lr=0.001):
        super(Net_struct, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder(size_in=encoder.size_in, size=encoder.size, depth=encoder.depth)    
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, prev):
        rep = self.encoder(x)
        pred = self.decoder(rep, prev)
        return pred.squeeze()

    def encode(self, expressions):
        return encode(self, expressions)    

    def loss(net, es, pad=10, length=None):
        if length is None:
            length = max([ len(list(e.infix())) for e in es ])
        data_in  = torch.tensor([ list(A.vocabify(e.infix(), length=length)) for e in es ], dtype=torch.long, device=get_device(net)) 
        tgt = torch.tensor([ list(A.vocabify(e.prefix(), length=length, pad_end=True)) for e in es ], dtype=torch.long, device=get_device(net)) 
        prev = make_prev(tgt, pad=pad)
        pred = net(data_in, prev)
        loss = F.cross_entropy(pred.view(pred.size(0)*pred.size(1),-1), tgt.view(tgt.size(0)*tgt.size(1)), ignore_index=pad)
        return loss        

    def experiment(net, batch_size=16, n=1000, interval=100, decay=1.5, pad=10, prefix=''):
        cost = Counter(dict(cost=0,N=0))
        for i in range(1,n+1):
            L, es = A.generate_batch(p=1.0, decay=decay, size=batch_size)
            loss = net.loss(es, pad=pad, length=L)
            net.opt.zero_grad()
            loss.backward()
            net.opt.step()
            cost += Counter(dict(cost=loss.data.item(), N=1))
            if i % interval == 0:
                print(prefix, i, loss.data.item(), cost['cost']/cost['N'])

def get_device(net):
    return list(net.parameters())[0].device
    
def make_prev(x, pad=10):
    return torch.cat([torch.zeros_like(x[:,0:1])+pad, x], dim=1)[:, :-1]

def encode_MLP(self, x):
    rep = self.encoder(x)
    pred = torch.tanh(self.MLP.L1(rep))
    return pred.squeeze()

def encode(net, expressions):
        L = max(len(list(e.infix())) for e in expressions)
        data_in =  torch.tensor([ list(A.vocabify(e.infix(), length=L)) for e in expressions], dtype=torch.long, device=get_device(net))
        return net.encoder(data_in).squeeze()   

def predict(net, expressions):
        L = max(len(list(e.infix())) for e in expressions)
        data_in =  torch.tensor([ list(A.vocabify(e.infix(), length=L)) for e in expressions], dtype=torch.long, device=get_device(net))
        return net(data_in).squeeze()   

def borrow_weights(source, target):
    for s, t in zip(source.parameters(), target.parameters()):
        t.data = s.data


    

