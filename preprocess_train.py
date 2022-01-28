# preprocess.py

import dataclasses

from sklearn.model_selection import train_test_split

from config import PreprocessConfig
from datasets import MeaningEmbeddingSentenceStyleDataset
from experiment import Experiment
from settings import EXPERIMENTS_DIR
from utils import save_pickle, load_pickle, load_embeddings, create_embeddings_matrix, extract_word_embeddings_style_dimensions
from vocab import Vocab


# train.py

import itertools
from collections import defaultdict

import numpy as np
import torch.utils.data
from ignite.engine import Engine, Events
from ignite.metrics import Metric
from tensorboardX import SummaryWriter

from config import TrainConfig
from experiment import Experiment
from models import Seq2Seq, Seq2SeqMeaningStyle, StyleClassifier
# from preprocess import load_dataset
# from settings import EXPERIMENTS_DIR
from update_functions import Seq2SeqUpdateFunction, StyleClassifierUpdateFunction, Seq2SeqMeaningStyleUpdateFunction
from utils import to_device, save_weights, init_weights
from vocab import Vocab


# preprocess.py

def save_dataset(exp, dataset_train, dataset_val, dataset_test, vocab, style_vocab, W_emb):
    # save_pickle((dataset_train, dataset_val, dataset_test), exp.experiment_dir.joinpath('datasets.pkl'))
    # save_pickle((vocab, style_vocab), exp.experiment_dir.joinpath('vocabs.pkl'))
    save_pickle(W_emb, exp.experiment_dir.joinpath('W_emb.pkl'))

    print(f'Saved: {exp.experiment_dir}')


def load_dataset(exp):
    # dataset_train, dataset_val, dataset_test = load_pickle(exp.experiment_dir.joinpath('datasets.pkl'))
    # vocab, style_vocab = load_pickle(exp.experiment_dir.joinpath('vocabs.pkl'))
    W_emb = load_pickle(exp.experiment_dir.joinpath('W_emb.pkl'))

    return dataset_train, dataset_val, dataset_test, vocab, style_vocab, W_emb


def create_dataset_reader(cfg):
    dataset_reader_class = cfg.dataset_reader_class

    dataset_reader_params = dataclasses.asdict(cfg)
    dataset_reader = dataset_reader_class(**dataset_reader_params)

    return dataset_reader


def create_vocab(instances):
    vocab = Vocab([Vocab.PAD_TOKEN, Vocab.START_TOKEN, Vocab.END_TOKEN, Vocab.UNK_TOKEN, ])
    vocab.add_documents([inst['sentence'] for inst in instances])

    style_vocab = Vocab()
    style_vocab.add_document([inst['style'] for inst in instances])

    return vocab, style_vocab


def create_splits(cfg, instances):
    if cfg.test_size != 0:
        instances_train_val, instances_test = train_test_split(instances, test_size=cfg.test_size, random_state=42)
    else:
        instances_test = []
        instances_train_val = instances

    if cfg.val_size != 0:
        instances_train, instances_val = train_test_split(instances_train_val, test_size=cfg.val_size, random_state=0)
    else:
        instances_train = []
        instances_val = []

    return instances_train, instances_val, instances_test


# train.py

def create_model(cfg, vocab, style_vocab, max_len, W_emb=None):
    model_class = cfg.model_class
    model_params = dataclasses.asdict(cfg)
    model_params.update(dict(
        max_len=max_len,
        vocab_size=len(vocab),
        start_index=vocab[Vocab.START_TOKEN],
        end_index=vocab[Vocab.END_TOKEN],
        pad_index=vocab[Vocab.PAD_TOKEN],
        nb_styles=len(style_vocab)
    ))

    if cfg.pretrained_embeddings:
        model_params.update(dict(
            W_emb=W_emb,
        ))

    model = model_class(**model_params)

    init_weights(model)

    model = to_device(model)

    return model


def create_update_function(cfg, model):
    update_function_class = None
    if isinstance(model, Seq2Seq):
        update_function_class = Seq2SeqUpdateFunction
    if isinstance(model, Seq2SeqMeaningStyle):
        update_function_class = Seq2SeqMeaningStyleUpdateFunction
    if isinstance(model, StyleClassifier):
        update_function_class = StyleClassifierUpdateFunction

    update_function_params = dataclasses.asdict(cfg)
    update_function_params.update(dict(
            model=model,
    ))

    update_function_train = update_function_class(train=True, **update_function_params)
    update_function_eval = update_function_class(train=False, **update_function_params)

    return update_function_train, update_function_eval

class LossAggregatorMetric(Metric):
    def __init__(self, *args, **kwargs):
        self.total_losses = defaultdict(float)
        self.num_updates = defaultdict(int)
        super().__init__(*args, **kwargs)

    def reset(self):
        self.total_losses = defaultdict(float)
        self.num_updates = defaultdict(int)

    def update(self, output):
        for name, val in output.items():
            self.total_losses[name] += float(val)
            self.num_updates[name] += 1

    def compute(self):
        losses = {name: val / self.num_updates[name] for name, val in self.total_losses.item()}

        return losses


def log_progress(epoch, iteration, losses, mode='train', tensorboard_writer=None, use_iteration=False):
    # print("function log_progress executed")
    if not use_iteration:
        losses_str = [
                f'{name}: {val:.3f}'
            for name, val in losses.items()
        ]
        losses_str = ' | '.join(losses_str)

        epoch_str = f'Epoch [{epoch}|{iteration}] {mode}'

        print(f'{epoch_str:<25}{losses_str}')

    for name, val in losses.items():
        tensorboard_writer.add_scalar(f'{mode}/{name}', val , epoch if not use_iteration else iteration)



def main(preprocess_cfg, train_cfg):
    with Experiment(EXPERIMENTS_DIR, preprocess_cfg, prefix='preprocess') as exp:
        print(f'Experiment started: {exp.experiment_id}')

        # read instances
        dataset_reader = create_dataset_reader(exp.config)
        print(f'Dataset reader: {dataset_reader.__class__.__name__}')

        instances = dataset_reader.read(exp.config.data_path)
        print(f'Instances: {len(instances)}')

        # create vocabularies
        vocab, style_vocab = create_vocab(instances)
        print(f'Vocab: {len(vocab)}, style vocab: {style_vocab}')

        if exp.config.max_vocab_size != 0:
            vocab.prune_vocab(exp.config.max_vocab_size)

        # create splits
        instances_train, instances_val, instances_test = create_splits(exp.config, instances)
        print(f'Train: {len(instances_train)}, val: {len(instances_val)}, test: {len(instances_test)}')

        # create embeddings
        word_embeddings = load_embeddings(preprocess_cfg)
        W_emb = create_embeddings_matrix(word_embeddings, vocab)

        # extract style dimensions
        style_dimensions = extract_word_embeddings_style_dimensions(preprocess_cfg, instances_train, vocab, style_vocab, W_emb)

        # create datasets
        dataset_train = MeaningEmbeddingSentenceStyleDataset(
            W_emb, style_dimensions, exp.config.style_tokens_proportion,
            instances_train, vocab, style_vocab
        )
        dataset_val = MeaningEmbeddingSentenceStyleDataset(
            W_emb, style_dimensions, exp.config.style_tokens_proportion,
            instances_val, vocab, style_vocab
        )
        dataset_test = MeaningEmbeddingSentenceStyleDataset(
            W_emb, style_dimensions, exp.config.style_tokens_proportion,
            instances_test, vocab, style_vocab
        )

        save_dataset(exp, dataset_train, dataset_val, dataset_test, vocab, style_vocab, W_emb)

        print(f'Experiment finished: {exp.experiment_id}')

    with Experiment(EXPERIMENTS_DIR, train_cfg, prefix='train') as exp:
        print(f'Experiment started: {exp.experiment_id}')

        #preprocessとtrainをまとめてるからこれはいらない
        # preprocess_exp = Experiment.load(EXPERIMENTS_DIR, exp.config.preprocess_exp_id)
        # dataset_train, dataset_val, dataset_test, vocab, style_vocab, W_emb = load_dataset(preprocess_exp)

        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=exp.config.batch_size, shuffle=True)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=exp.config.batch_size, shuffle=False)
        print(f'Data loader : {len(data_loader_train)}, {len(data_loader_val)}')

        model = create_model(exp.config, vocab, style_vocab, dataset_train.max_len, W_emb)

        update_function_train, update_function_eval = create_update_function(exp.config, model)

        trainer = Engine(update_function_train)
        evaluator = Engine(update_function_eval)

        metrics = {'loss': LossAggregatorMetric(), }
        for metric_name, metric in metrics.items():
            metric.attach(evaluator, metric_name)

        best_loss = np.inf

        print("preprocess completed")

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_iter(engine):
            losses_train = engine.state.output
            log_progress(trainer.state.epoch, trainer.state.iteration, losses_train, 'train', tensorboard_writer, True)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            nonlocal best_loss

            # evaluator.run(data_loader_train)
            # losses_train = evaluator.state.metrics['loss']

            evaluator.run(data_loader_val)
            losses_val = evaluator.state.metrics['loss']

            # log_progress(trainer.state.epoch, trainer.state.iteration, losses_train, 'train', tensorboard_writer)
            log_progress(trainer.state.epoch, trainer.state.iteration, losses_val, 'val', tensorboard_writer)

            if losses_val[exp.config.best_loss] < best_loss:
                best_loss = losses_val[exp.config.best_loss]
                save_weights(model, exp.experiment_dir.joinpath('best.th'))

        tensorboard_dir = exp.experiment_dir.joinpath('log')
        tensorboard_writer = SummaryWriter(str(tensorboard_dir))

        trainer.run(data_loader_train, max_epochs=exp.config.num_epochs)

        print(f'Experiment finished: {exp.experiment_id}')



if __name__ == '__main__':
    main(PreprocessConfig(), TrainConfig())
