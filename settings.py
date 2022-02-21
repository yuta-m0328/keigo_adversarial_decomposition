from pathlib import Path

DATA_DIR = Path('./data/')
EXPERIMENTS_DIR = DATA_DIR.joinpath('experiments/')

SHAKESPEARE_DATASET_DIR = DATA_DIR.joinpath('datasets/shakespeare/data/align/plays/merged/')
YELP_DATASET_DIR = DATA_DIR.joinpath('datasets/yelp/data/yelp')
# KEIGO_DATASET_DIR = DATA_DIR.joinpath('datasets/keigo/practice_addecom_datasets')
KEIGO_DATASET_DIR = DATA_DIR.joinpath('datasets/keigo/addecom_datasets')

WORD_EMBEDDINGS_FILENAMES = dict(
    # glove=DATA_DIR.joinpath('word_embeddings/glove.840B.300d.pickled'),
    glove=DATA_DIR.joinpath('word_embeddings/glove.6B.300d.magnitude'),
    fast_text=DATA_DIR.joinpath('word_embeddings/crawl-300d-2M.pickled'),
    gensim=DATA_DIR.joinpath('word_embeddings/cc.ja.300.vec.gz'),
    # magnitude=DATA_DIR.joinpath('word_embeddings/chive-1.2-mc90.magnitude'),
    magnitude=DATA_DIR.joinpath('word_embeddings/cc.ja.300.magnitude')
    
)
