import pickle

import numpy as np
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained('../pretrained_model/bart-base')

# 读取npy文件
node_embeddings = np.load("./node_embeddings.npy")
with open('./node.pkl', 'rb') as file:
    node = pickle.load(file)
with open('./nodeidx2bartidx.pkl', 'rb') as file:
    nodeidx2bartidx = pickle.load(file)
print("0")

