import pickle
import json

import copy
import re
import unidecode
import random
import spacy



class NLP:
    def __init__(self):
        # self.nlp = spacy.load('../../../en_core_web_sm-2.3.0', disable=['ner', 'parser', 'tagger'])
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def sent_tokenize(self, text):
        doc = self.nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None:
            return text
        text = ' '.join(text.split())
        if lower:
            text = text.lower()
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        return ' '.join(toks)


nlp = NLP()


def get_neibornode(i, all_sub, subgraph, data):
    i = i + 1
    # temp_sub = subgraph
    for tri in data['triples'][i:]:
        temp_sub = copy.deepcopy(subgraph)
        h = data['nodes'].index(tri[0])
        r = data['nodes'].index(tri[1])
        t = data['nodes'].index(tri[2])
        if h in temp_sub or t in temp_sub:
            temp_sub.add(h)
            temp_sub.add(r)
            temp_sub.add(t)
            all_sub.append(list(temp_sub))
            # all_sub.add(tuple(temp_sub))
            get_neibornode(data['triples'].index(tri), all_sub, temp_sub, data)


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    d = [m.group(0) for m in matches]
    new_d = []
    for token in d:
        token = token.replace('(', '')
        token_split = token.split('_')
        for t in token_split:
            new_d.append(t)
    new_d = " ".join(new_d)
    return new_d


def get_nodes(n):
    n = unidecode.unidecode(n.strip().lower())
    n = n.replace('-', ' ')
    n = n.replace('_', ' ')
    n = n.strip('\"')
    n = nlp.word_tokenize(n)

    return n


def get_relation(n):
    n = unidecode.unidecode(n.strip().lower())
    n = n.replace('-', ' ')
    n = n.replace('_', ' ')
    n = nlp.word_tokenize(n)

    return n


def get_text(txt, lower=True):
    if lower:
        txt = txt.lower()
    txt = unidecode.unidecode(txt.strip())
    txt = txt.replace('-', ' ')
    txt = nlp.word_tokenize(txt)

    return txt


all_entry = open('test.cPickle', 'rb')
data = pickle.load(all_entry)

fout = open("test.json", "w", encoding='utf-8')

for entry in data:
    for lex in entry.lexEntries:
        new_dict = dict()
        valid = True
        ner_dict = {}
        ent_dict = {}
        tags = []
        for re in lex.references:
            tags.append(re.tag)
            en = get_nodes(re.entity)
            tag = re.tag.replace("-", "_")
            ner_dict[tag] = en
            ent_dict[en] = tag
        new_dict['ner2ent'] = ner_dict

        # new_dict['target'] = lex.template
        # new_dict['target_txt']=lex.text
        new_dict['triples'] = []
        for triple_list in lex.orderedtripleset:
            for triple in triple_list:
                new_triple = [get_nodes(triple.subject), get_relation(triple.predicate),
                              get_nodes(triple.object)]  # [h,r,t]
                new_dict['triples'].append(new_triple)

        tokens = []
        print(lex.template)
        template = lex.template
        for tag in tags:
            template = template.replace(tag, tag.replace('-', '_'))
        print(template)
        for token in template.split():
            if token.isupper() and '_' in token:  # 包含下换线和全是大写的字符串
                tokens.append(token)
            else:
                tokens.append(token.lower())
        new_dict['target'] = get_text(' '.join(tokens), lower=False)  # 拼成一句话

        try:
            tokens = []
            entities = []
            for token in new_dict['target'].split():
                if token.isupper():
                    tokens.append(new_dict['ner2ent'][token])  # ner转换为entity
                    if new_dict['ner2ent'][token] not in entities:
                        entities.append(new_dict['ner2ent'][token])
                else:
                    tokens.append(token)
            new_dict['target_txt'] = (' '.join(tokens)).lower()
        except KeyError:
            continue

        fout.write(json.dumps(new_dict, ensure_ascii=False) + "\n")
fout.close()
