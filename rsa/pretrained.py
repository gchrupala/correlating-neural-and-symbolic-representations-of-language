import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import nltk
from infersent.models import InferSent
import os.path


def encode_bert(texts, trained=True, large=False):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') \
            if large else  BertTokenizer.from_pretrained('bert-base-uncased')
    if trained:
        model = BertModel.from_pretrained('bert-large-uncased') \
            if large else BertModel.from_pretrained('bert-base-uncased')
    else:
        model = BertModel(BertModel.from_pretrained('bert-large-uncased').config \
            if large else BertModel.from_pretrained('bert-base-uncased').config)
    model.eval()
    for text in texts:
        text = "[CLS] {} [SEP]".format(text.lower())
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            encoded_layers, pooled = model(tokens_tensor)
            yield encoded_layers, pooled

def prepare_bert(params, texts):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') \
            if params['bert']['large'] else  BertTokenizer.from_pretrained('bert-base-uncased')
    if params['bert']['trained']:
        model = BertModel.from_pretrained('bert-large-uncased') \
            if params['bert']['large'] else BertModel.from_pretrained('bert-base-uncased')
    else:
        model = BertModel(BertModel.from_pretrained('bert-large-uncased').config \
            if params['bert']['large'] else BertModel.from_pretrained('bert-base-uncased').config)
    model.eval()
    params['encoder'] = model
    params['tokenizer'] = tokenizer

def batcher_bert(params, texts):
    model = params['encoder']
    tokenizer = params['tokenizer']
    emb = []
    for text in [' '.join(text) for text in texts ]:
        text = "[CLS] {} [SEP]".format(text.lower())
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor)
            emb.append(encoded_layers[params['bert']['layer']][0,0,:].cpu().numpy())
    return np.vstack(emb)

def encode_infersent(texts, trained=True, tokenize=True):
    from infersent.models import InferSent
    V = 2
    MODEL_PATH = 'infersent/encoder/infersent%s.pickle' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    W2V_PATH = 'infersent/dataset/fastText/crawl-300d-2M-subword.vec'
    if trained:
        logging.info("Loading pretrained weights")
        infersent.load_state_dict(torch.load(MODEL_PATH))  
        infersent.set_w2v_path(W2V_PATH)
    else:
        RANDOM_PATH = "infersent/dataset/fastText/random.vec"
        if not os.path.isfile(RANDOM_PATH):
            random_embeddings(W2V_PATH, RANDOM_PATH)
            logging.info("Written random word embeddings to file")
        infersent.set_w2v_path(RANDOM_PATH)        
    infersent.build_vocab(texts, tokenize=tokenize)
    return torch.tensor(infersent.encode(texts, tokenize=tokenize))

def prepare_infersent(params, texts):
    from infersent.models import InferSent
    texts = [ ' '.join(text) for text in texts ]
    V = 2
    MODEL_PATH = 'infersent/encoder/infersent%s.pickle' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    W2V_PATH = 'infersent/dataset/fastText/crawl-300d-2M-subword.vec'
    if params['infersent']['trained']:
        logging.info("Loading pretrained weights")
        infersent.load_state_dict(torch.load(MODEL_PATH))  
        infersent.set_w2v_path(W2V_PATH)
    else:
        RANDOM_PATH = "infersent/dataset/fastText/random.vec"
        if not os.path.isfile(RANDOM_PATH):
            random_embeddings(W2V_PATH, RANDOM_PATH)
            logging.info("Written random word embeddings to file")
        infersent.set_w2v_path(RANDOM_PATH)
    infersent.build_vocab(texts, tokenize=False)
    params['encoder'] = infersent

def batcher_infersent(params, texts):
    sent = [' '.join(text) for text in texts ]
    params['encoder'].encode(sent, bsize=params['batch_size'], tokenize=False)

def random_embeddings(path, target_path):
    D ={}
    with open(path) as f:
        for i,line in enumerate(f):
            if i > 0:
                word, vec = line.split(' ', 1)
                D[word] = np.fromstring(vec, sep=' ')
    logging.info("Loaded word embeddings")
    w = np.concatenate([[row] for row in D.values()], axis=0)
    loc = w.mean(axis=0)
    scale = w.std(axis=0)
    rows = np.random.normal(loc=loc, scale=scale, size=w.shape) 
    logging.info("Generated random embeddings")
    with open(target_path, 'w') as f:
        f.write("2000000 300\n")
        for k,r in zip(D.keys(), rows):
            f.write("{} {}\n".format(k, " ".join(["{:.4}".format(n) for n in r])))

