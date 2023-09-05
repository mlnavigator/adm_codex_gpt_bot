import json
import os
import re
from functools import lru_cache
import pickle
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from transformers import AutoTokenizer, AutoModel

import pymorphy2
morph = pymorphy2.MorphAnalyzer(lang='ru')


base_dir = os.path.dirname(os.path.abspath(__file__))

fn = os.path.join(base_dir, 'assets/data.json')

with open(fn, 'r') as f:
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


texts_cleared = [clear_text(t[2]) for t in data]

texts_tokenized = [[lemmatize(w) for w in t.split()] for t in texts_cleared]

stop_words = """или, но, дабы, затем, потом, лишь только, он, мы, его, вы, вам, вас, ее, что, 
который, их, все, они, я, весь, мне, меня, таким, для, на, по, со, из, от, до, без, над, под, за, при, после, во,
же, то, бы, всего, итого, даже, да, нет, ой, ого, эх, браво, здравствуйте, спасибо, извините,
скажем, может, допустим, честно говоря, например, на самом деле, однако, вообще, в, общем, вероятно, очень, 
минимально, максимально, абсолютно, огромный, предельно, сильно, слабо, самый, сайт, давать, всегда, однако, и, а, но, да, если, что, когда, потому, что, так, как, как, будто, 
вследствие, того, что, с, тех, пор, как, в, то, время, как, для, того, чтобы, ни, то, ли, но, зато, от, и, к, кто, что, 
такой, такое, такая, почему"""

stop_words_1  = stop_words.replace('\n','').split(', ')

texts_tokenized_1 = [' '.join([tt for tt in t if len(tt) > 1 and tt not in stop_words_1]) for t in texts_tokenized]

# start creating tfidf vects
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts_tokenized_1)

with open(os.path.join(base_dir, 'assets/tfidf_sklearn_vectoryzer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)
    
with open(os.path.join(base_dir, 'assets/tfidf_vects.pkl'), 'wb') as f:
    pickle.dump(X, f)
    
print('tf_idf vects done')

# start createing bert embeddings
tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

model.to('cpu')
model.eval()


def embed_bert_cls(texts, model=model, tokenizer=tokenizer):
    t = tokenizer(texts, max_length=512, padding=True, 
                  truncation=True, return_tensors='pt', add_special_tokens=True)
    
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    
    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings.cpu().numpy()


embs_bert = []

for t in tqdm(data):
    emb = embed_bert_cls(t[2])[0]
    embs_bert.append(emb)

embs_bert = np.vstack(embs_bert)
embs_bert = embs_bert.astype(np.float16)

with open(os.path.join(base_dir, 'assets/bert_vects.pkl'), 'wb') as f:
    pickle.dump(embs_bert, f)

print('bert vects done', embs_bert.shape)
print('vectoryzing finished')
