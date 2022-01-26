# Adversarial Decomposition of Text Representation
The code for the paper "Adversarial Decomposition of Text Representation", NAACL 2019 
https://arxiv.org/abs/1808.09042

# Installation 

 1. Clone this repo: `https://github.com/itachicom615/keigo_adversarial_decomposition.git`
 2. Install PyTorch v1.1.0: `pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl` (for python3.6, use `pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl`)
 3. Install dependencies: `pip install -r requirements.txt`
 4. Download spacy models: `python -m spacy download ja_core_news_lg==3.2.0`
 5. Download gensim: `pip install gensim==3.8.3`

# Initial setup

 1. Create dir `mkdir -p data/experiments`
 2. Create dir `mkdir -p data/datasets`
 3. Create dir `mkdir -p data/word_embeddings`
 4. Download the Keigo data: `git clone https://github.com/itachicom615/keigo_extraction.git data/datasets/keigo`
 5. Download the pickled fastText embeddings `wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz data/word_embeddings`

# Running the code

Global constants are set in the file `settings.py`. In general, you don't need to change this file.
Experiment parameters are set in the `config.py` file. 

First, run the preprocessing script: `python preprocess_train.py`

This scirpt will print the ID of the training experiment. You can paste it in the `eval_generation.ipynb` notebook to play with the model.

## Chaning the form and meaning
The provided `eval_generation.ipynb` notebook shows how to use the model to swap the meaning and form vectors of the input sentences!


# Citation
If you find this code helpful, please consider citing our paper:

...

