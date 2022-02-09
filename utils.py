import itertools
import json
import pickle
import joblib
import dill
import re

import numpy as np
import torch

import gensim
from pymagnitude import Magnitude #Magnitudeで行いたい（速いので)

from settings import WORD_EMBEDDINGS_FILENAMES
from vocab import Vocab


def save_json(obj, filename):
    with open(filename, 'wb') as f:
        json.dump(obj, f)


def load_json(filename):
    with open(filename, 'rb') as f:
        obj = json.load(f)

    return obj


def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj,f)
        # pickle.dump(str(obj), f)
        # print("checkpoint1")
        # print(f'obj:{obj}')
        # print(f'obj type:{type(obj)}')
        # if type(obj) is tuple:
            # for i in obj:
                # print(f'checkpoint1-{i}')
                # print(f'obj{i}:{obj}')
                #print(f'obj type{i}:{type(obj)}')
            # pickle.dump(str(obj), f)
            # pickle.dump(obj, f)
            # print(obj)
            # return
        # else:
            # print("obj is not tuple")
            # pickle.dump(str(obj), f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        return obj
        # print("checkpoint2")
        # print(f'obj:{obj}')
        # print(f'obj type:{type(obj)}')
        # if type(obj) is str:
        #     return obj
        # else:
        #     return obj
        

def save_dill(obj, filename):
    with open(filename, 'wb') as f:
        f.write(obj)

def load_dill(filename):
    with open(filename, 'rb') as f:
        obj = dill.load(f)

    return obj


def to_device(obj, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(obj, (list, tuple)):
        return [to_device(o, device) for o in obj]

    if isinstance(obj, dict):
        return {k: to_device(o, device) for k, o in obj.items()}

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = obj.to(device)
    return obj


def load_weights(model, filename):
    # load trained on GPU models to CPU
    if not torch.cuda.is_available():
        def map_location(storage, loc): return storage
    else:
        map_location = None

    state_dict = torch.load(str(filename), map_location=map_location)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.load_state_dict(state_dict)


def save_weights(model, filename):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), str(filename))


def init_weights(modules):
    if isinstance(modules, torch.nn.Module):
        modules = modules.modules()

    for m in modules:
        if isinstance(m, torch.nn.Sequential):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.ModuleList):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
            torch.nn.init.xavier_normal_(m.weight.data)
            # m.bias.data.zero_()
            if m.bias is not None:
                m.bias.data.normal_(0, 0.01)

        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()


def get_sequences_lengths(sequences, masking=0, dim=1):
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2)

    masks = torch.ne(sequences, masking).long()

    lengths = masks.sum(dim=dim)

    return lengths.cpu() #tensorをcpuにしないとpack_pad_sequenceでエラーが出る


def load_embeddings(cfg):
    word_embeddings_filename = WORD_EMBEDDINGS_FILENAMES[cfg.word_embeddings]
    if cfg.word_embeddings == 'gensim':
        print(f"use {cfg.word_embeddings} word embeddings.")
        # word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(word_embeddings_filename, binary=False)
        word_embeddings  = Magnitude(word_embeddings_filename)
    else:
        word_embeddings = load_pickle(word_embeddings_filename)
    

    return word_embeddings


def create_embeddings_matrix(word_embeddings, vocab):
    # print(word_embeddings)
    #　ここは埋め込みサイズ(=300次元)がわかればいい
    # embedding_size = word_embeddings[list(word_embeddings.keys())[0]].shape[0]
    # embedding_size = word_embeddings.vector_size #gensim
    embedding_size = word_embeddings[0][1].shape[0] #magnitude
    # print(f"utils.py word_embeddings v")
    # print(word_embeddings)

    W_emb = np.zeros((len(vocab), embedding_size), dtype=np.float32)
    special_tokens = {
        t: np.random.uniform(-0.3, 0.3, (embedding_size,))
        for t in (Vocab.START_TOKEN, Vocab.END_TOKEN, Vocab.UNK_TOKEN)
    }
    
    # print("start W_emb vvv")
    # print(W_emb)

    special_tokens[Vocab.PAD_TOKEN] = np.zeros((embedding_size,))
    nb_unk = 0
    # keys = [key for key,value in word_embeddings] #magnitude
    with open("data/magnitude_keys","rb") as f:
        keys = pickle.load(f)
    for i, t in vocab.id2token.items():
        # print(i,t)
        if t in special_tokens:
            W_emb[i] = special_tokens[t]
        else:
            # if t in word_embeddings:
                # W_emb[i] = word_embeddings[t] #gensim
                # print(f"utils.py c_matrix => word_embeddings[t] : {word_embeddings[t]}")
            #if t.text in keys: #magnitude
            if t in keys: #spacy要素を排除
                W_emb[i] = word_embeddings[i][1] #magnitude
            else:
                W_emb[i] = np.random.uniform(-0.3, 0.3, embedding_size)
                nb_unk += 1

    # print(f'Nb unk: {nb_unk}')
    # print(f'W_emb vvv')
    # print(W_emb)

    return W_emb


def extract_word_embeddings_style_dimensions(cfg, instances, vocab, style_vocab, W_emb):
    sample_size = min(cfg.nb_style_dims_sentences, len(instances))
    instances = np.random.choice(instances, size=sample_size, replace=False)
    instances_grouped_by_style = [
        [inst['sentence'] for inst in instances if inst['style'] == style]
        for style in style_vocab.token2id.keys()
    ]
    print(f'Styles instances: {[len(s) for s in instances_grouped_by_style]}')

    sentences_embed = [
        [
            W_emb[vocab[t]]
            for t in itertools.chain.from_iterable(style_sents)
            if t in vocab
        ]
        for style_sents in instances_grouped_by_style
    ]

    means = [np.mean(e, axis=0) for e in sentences_embed]
    print(f'Styles means: {[m.shape for m in means]}')

    # get dimensions that have the biggest absolute difference
    means_diff = np.abs(np.subtract(*means))
    diff_sort_idx = np.argsort(-means_diff)
    style_dims = diff_sort_idx[:cfg.nb_style_dims]

    print(f'Style dimensions: {style_dims.shape}')

    return style_dims
