seed=100
import random
random.seed(seed)
import torch
torch.manual_seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
import pickle
import json
import rsa.arithmetic as A
import rsa.correlate as C
import rsa.models as M
import rsa.experiment as E
import rsa.kernel as K

config = dict(batch_size=32,
              n=10000,
              N=1000,
              epochs=[10, 20, 40],
              decay=[2.0, 1.8, 1.5],
              size=128,
              depth=1,
              size_embed=64,
              n_classes=None)
value_f = lambda e: e.evaluate(10)
_, data = A.generate_batch(p=1.0, decay=config['decay'][-1], size=config['N']) # Eval data with target decay
pickle.dump(data, open('data.pkl','wb'))
scorer = E.Scorer(data, depth_f=lambda e: e.depth(), 
                     value_f=lambda e: e.evaluate(10), 
                     kernel_f=lambda E: K.pairwise(E, normalize=True, ignore_leaves=True), 
                     device='cuda')
net = M.Net(M.Encoder(size_in=15, size=config['size'], depth=config['depth'], size_embed=config['size_embed']), n_classes=config['n_classes'], target_type='value').cuda()
for i, decay in enumerate(config['decay']):
    for epoch in range(0,config['epochs'][i]+1):
        if epoch>0:
            net.experiment(batch_size=config['batch_size'], n=config['n'], decay=decay, prefix="{} {}".format(epoch, decay))
        pickle.dump(net, open('net-d{}-{:02d}.pkl'.format(decay,epoch), 'wb'))
        result = scorer.score(lambda E: net.encode(E))
        json.dump(result, open('result-d{}-{:02d}.json'.format(decay,epoch), 'w'), indent=2)

    
