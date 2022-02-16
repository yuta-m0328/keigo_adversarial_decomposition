import random

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix

from settings import EXPERIMENTS_DIR
from experiment import Experiment
from utils import to_device, load_weights, load_embeddings, create_embeddings_matrix
from vocab import Vocab
from train import create_model
from preprocess_train import load_dataset, create_dataset_reader



def create_inputs(instances, dataset_train, dataset_reader, style_vocab):
    if not isinstance(instances, list):
         instances = [instances,]

    if not isinstance(instances[0], dict):
        sentences = [
            dataset_reader.preprocess_sentence(dataset_reader.spacy( dataset_reader.clean_sentence(sent)))
            for sent in instances
        ]
        
        style = list(style_vocab.token2id.keys())[0]
        instances = [
            {
                'sentence': sent,
                'style': style,
            }
            for sent in sentences
        ]
        
        for inst in instances:
            inst_encoded = dataset_train.encode_instance(inst)
            inst.update(inst_encoded)

    instances = [
        {
            'sentence': inst['sentence_enc'],
            'style': inst['style_enc'],
        } 
        for inst in instances
    ]
    
    instances = default_collate(instances)
    instances = to_device(instances)      
    
    return instances

def get_sentences(outputs, vocab):
    predicted_indices = outputs["predictions"]
    end_idx = vocab[Vocab.END_TOKEN]
    
    if not isinstance(predicted_indices, np.ndarray):
        predicted_indices = predicted_indices.detach().cpu().numpy()

    all_predicted_tokens = []
    for indices in predicted_indices:
        indices = list(indices)

        # Collect indices till the first end_symbol
        if end_idx in indices:
            indices = indices[:indices.index(end_idx)]

        predicted_tokens = [vocab.id2token[x] for x in indices]
        all_predicted_tokens.append(predicted_tokens)
    
    return all_predicted_tokens

def show_sentence(target, model, dataset_train, dataset_val, dataset_reader, vocab ,style_vocab):
    print('関数show_sentenceを実行します')
    sentence = ' '.join(dataset_val.instances[target]['sentence'])
    print(f'dataset_valの{target}文目の文章を表示します')
    print(sentence)

    inputs = create_inputs(sentence, dataset_train, dataset_reader, style_vocab)
    outputs = model(inputs)
    sentences = get_sentences(outputs, vocab)
    print(f'上の文章を変換したものを表示します')
    ' '.join(sentences[0])
    print()

def swap_style(style_vocab, dataset_val, num):
    print('関数swap_styleを実行します')
    possible_styles = list(style_vocab.token2id.keys())
    print('styleの種類を表示します')
    print(possible_styles)

    sentences0 = [s for s in dataset_val.instances if s['style'] == possible_styles[0]]
    sentences1 = [s for s in dataset_val.instances if s['style'] == possible_styles[1]]

    print(f'{possible_styles[0]}の文章をランダムに{num}つ表示します')
    for i in np.random.choice(np.arange(len(sentences0)), num):
        print(i, ' '.join(sentences0[i]['sentence']))
    print()

    print(f'{possible_styles[1]}の文章をランダムに{num}つ表示します')
    for i in np.random.choice(np.arange(len(sentences1)), num):
        print(i, ' '.join(sentences1[i]['sentence']))
    print()

def swap(dataset_train, dataset_val, dataset_reader, vocab, style_vocab, model, target0, target1):
    print('関数swapを実行します')
    possible_styles = list(style_vocab.token2id.keys())

    sentences0 = [s for s in dataset_val.instances if s['style'] == possible_styles[0]]
    sentences1 = [s for s in dataset_val.instances if s['style'] == possible_styles[1]]

    print('入力された文です')
    print(' '.join(sentences0[target0]['sentence']))
    print(' '.join(sentences1[target1]['sentence']))
    print()

    inputs = create_inputs([
        sentences0[target0],
        sentences1[target1],
    ], dataset_train ,dataset_reader, style_vocab)

    z_hidden = model(inputs)
    print('style側隠れ層のサイズです')
    print(z_hidden['style_hidden'].shape)
    print('meaning側隠れ層のサイズです')
    print(z_hidden['meaning_hidden'].shape)
    print()
    original_decoded = model.decode(z_hidden)
    original_sentences = get_sentences(original_decoded, vocab)
    # print(' '.join(original_sentences[0]))
    # print(' '.join(original_sentences[1]))

    z_hidden_swapped = {
        'meaning_hidden': torch.stack([
            z_hidden['meaning_hidden'][0].clone(),
            z_hidden['meaning_hidden'][1].clone(),        
        ], dim=0),
        'style_hidden': torch.stack([
            z_hidden['style_hidden'][1].clone(),
            z_hidden['style_hidden'][0].clone(),        
        ], dim=0),
    }

    swaped_decoded = model.decode(z_hidden_swapped)
    swaped_sentences = get_sentences(swaped_decoded, vocab)
    print('original')
    print(' '.join(original_sentences[0]))
    print(' '.join(original_sentences[1]))
    print('swaped')
    print(' '.join(swaped_sentences[0]))
    print(' '.join(swaped_sentences[1]))
    print()

def t_SNE_visualization(dataset_train, dataset_val, dataset_reader, style_vocab, model, num, path):
    print('関数t_SNE_visualizationを実行します')
    # sentence listを作成する
    sentences = [" ".join(line['sentence']) for line  in dataset_val.instances[:num]]
    # gold(style) label listを作成する
    gold = [line['style'] for line  in dataset_val.instances[:num]]

    # sentences = []
    # gold = []
    # for i in range(1000):
    #     line = dataset_val.instances[i]
    #     if i != '<unk>':
    #         sentences.append(" ".join(line['sentence']))
    #         gold.append(line['style'])

    df = pd.DataFrame({0 : pd.Series(gold),
                       1 : pd.Series(sentences)})
    print('styleとsentenceのリストを表示します')
    print(df.head(10))

    # modelの出力からstyle embeddingを取り出す
    inputs_features = []
    for sentence in df[1]:
        if sentence != '<unk>':
            inputs_features.append(create_inputs(sentence, dataset_train, dataset_reader, style_vocab))
    # inputs_features = [create_inputs(sentence, dataset_train, dataset_reader, style_vocab) for sentence in df[1]]
    # outputs_features = []
    # for line in inputs_features:
    #     outputs = model(line)
    #     outputs_features.append(outputs)
    #     line = Variable(line, volatile=True)
    outputs_features = [model(line) for line in inputs_features]
    
    # style_embedを抽出する
    style_embed = [line['style_hidden'].to('cpu').detach().numpy().copy() for line  in outputs_features]

    # ndarrayの2次元配列を3次元配列にする
    style_embeds = np.array(style_embed)
    print('style_embedsのサイズです')
    print(style_embeds.shape)
    style_embeds_sq=np.squeeze(style_embeds)
    print('style_embeds_sqのサイズです')
    print(style_embeds_sq.shape)

    # ndarrayからcsr?matrixへの変換
    style_embed_csr =csr_matrix(style_embeds_sq)
    tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
    X_style = tsne.fit_transform(style_embeds_sq)
    ddf = pd.concat([df, pd.DataFrame(X_style, columns = ['col1', 'col2'])], axis = 1)
    gold_list = ddf[0].unique()
    print('gold_listの中身です')
    print(gold_list)

    # 可視化
    colors =  ["r", "g"]
    plt.figure(figsize = (30, 30))
    for i , v in enumerate(gold_list):
        tmp_df = ddf[ddf[0] == v]
        plt.scatter(tmp_df['col1'],  
                    tmp_df['col2'],
                    label = v,
                    color = colors[i])
        plt.legend(fontsize = 30)
        plt.savefig(path, format='png', bbox_inches = 'tight')

def main():
    exp_id ='./train.izi1kk0q' # edit id
    
    exp = Experiment.load(EXPERIMENTS_DIR, exp_id)
    print(exp.config.preprocess_exp_id)

    preprocess_exp = Experiment.load(EXPERIMENTS_DIR, exp.config.preprocess_exp_id)
    dataset_train, dataset_val, dataset_test, vocab, style_vocab, W_emb = load_dataset(preprocess_exp)

    dataset_reader = create_dataset_reader(preprocess_exp.config)

    model = create_model(exp.config, vocab, style_vocab, dataset_train.max_len, W_emb)

    load_weights(model, exp.experiment_dir.joinpath('best.th'))

    model = model.eval()

    show_sentence(100 ,model, dataset_train, dataset_val, dataset_reader, vocab, style_vocab)

    swap_style(style_vocab, dataset_val, 10)
    
    swap(dataset_train, dataset_val, dataset_reader, vocab, style_vocab, model, 1000, 2000)

    t_SNE_visualization(dataset_train, dataset_val, dataset_reader, style_vocab, model, 100, 'keigo_form_embeddings.png')


if __name__ == '__main__':
    main()
