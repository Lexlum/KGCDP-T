import torch
import pickle
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained('../pretrained_model/bart-base')
tokenizer = BartTokenizer.from_pretrained('../pretrained_model/bart-base')
embedding = model.get_input_embeddings().weight

vocab = pickle.load(open("relation.pkl", "rb"))

my_embedding = []
my_iddx = set()
for token, idx in vocab.items():
    rel_embedding = torch.zeros_like(embedding[0])
    for bart_token_idx in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token)):
        rel_embedding = rel_embedding+embedding[bart_token_idx]
    my_embedding.append(rel_embedding)
my_embedding = torch.stack(my_embedding, dim=0).detach().numpy()
np.save("relation_embeddings.npy", my_embedding)


vocab = pickle.load(open("node.pkl", "rb"))
my_embedding = []
my_iddx = set()
for token, idx in vocab.items():
    rel_embedding = torch.zeros_like(embedding[0])
    for bart_token_idx in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token)):
        rel_embedding = rel_embedding+embedding[bart_token_idx]
    my_embedding.append(rel_embedding)
my_embedding = torch.stack(my_embedding, dim=0).detach().numpy()
np.save("node_embeddings.npy", my_embedding)