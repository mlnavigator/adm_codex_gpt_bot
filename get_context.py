# extract data context according to user question
import json
import os
import re
from functools import lru_cache

import numpy as np
import pickle

import pymorphy2
morph = pymorphy2.MorphAnalyzer(lang='ru')

import torch
from transformers import AutoTokenizer, AutoModel


base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_fn = os.path.join(base_dir, 'assets/data.json')

with open(dataset_fn, 'r') as f:
    data = json.load(f)
    

@lru_cache(10000)
def lemmatize(s):
    s = str(s).lower()
    return morph.parse(s)[0].normal_form


def clear_text(t):
    t = str(t).lower()
    t = re.sub('[^\w]', ' ', t)
    t = re.sub('[\d]', ' ', t)
    t = re.sub('\s+', ' ', t)
    t = t.strip()
    t = ' '.join([w for w in t.split() if len(w) > 2])
    return t


stop_words = """или, но, дабы, затем, потом, лишь только, он, мы, его, вы, вам, вас, ее, что, 
который, их, все, они, я, весь, мне, меня, таким, для, на, по, со, из, от, до, без, над, под, за, при, после, во,
же, то, бы, всего, итого, даже, да, нет, ой, ого, эх, браво, здравствуйте, спасибо, извините,
скажем, может, допустим, честно говоря, например, на самом деле, однако, вообще, в, общем, вероятно, очень, 
минимально, максимально, абсолютно, огромный, предельно, сильно, слабо, самый, сайт, давать, всегда, однако, и, а, но, да, если, что, когда, потому, что, так, как, как, будто, 
вследствие, того, что, с, тех, пор, как, в, то, время, как, для, того, чтобы, ни, то, ли, но, зато, от, и, к, кто, что, 
такой, такое, такая, почему"""

stop_words_1  = stop_words.replace('\n','').split(', ')

with open(os.path.join(base_dir, 'assets/tfidf_sklearn_vectoryzer.pkl'), 'rb') as f:
    tf_idf_vectorizer = pickle.load(f)
    
with open(os.path.join(base_dir, 'assets/tfidf_vects.pkl'), 'rb') as f:
    tfidf_vects = pickle.load(f)

with open(os.path.join(base_dir, 'assets/bert_vects.pkl'), 'rb') as f:
    bert_vects = pickle.load(f)
    
bert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
bert_model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
bert_model.eval()


def prepare_q(q):
    q = str(q).lower()
    q = clear_text(q)
    q1 = [lemmatize(w) for w in q.split()]
    q1 = ' '.join([w for w in q1 if len(w) > 1 and w not in stop_words_1])
    return q1  


def find_similar_tfidf(q, n=10):
    global tf_idf_vectorizer
    global tfidf_vects
    q_vect = tf_idf_vectorizer.transform([prepare_q(q)])
    scores = tfidf_vects.dot(q_vect.T).toarray().flatten()
    if max(scores) == 0:
        return []
    ind = np.argsort(-scores)[:n]
    return list(zip(ind, scores[ind]))


def bert_vec(texts, model=bert_model, tokenizer=bert_tokenizer):
    t = tokenizer(texts, max_length=512, padding=True, 
                  truncation=True, return_tensors='pt', add_special_tokens=True)
    
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    
    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings.cpu().numpy()


def find_similar_bert(q, n=20):
    global bert_vects
    q_vect = bert_vec(q)[0]
    scores = np.matmul(bert_vects, q_vect).flatten()
    ind = np.argsort(-scores)[:n]
    return list(zip(ind, scores[ind]))


def get_context(q):
    sim_tfidf = find_similar_tfidf(q, n=5)
    sim_bert = find_similar_bert(q, n=5)
    
    res_ind = [i[0] for i in sim_bert]
    
    for i in sim_tfidf:
        if i[0] not in res_ind:
            res_ind.append(i[0])
            
    max_len = 10000
    context = []
    for i in res_ind:
        t = data[i]
        if len(context) + len(t[2]) <= max_len:
            context.append(t)
    return context
