from tqdm import tqdm
from config import PreprocessConfig
from utils import load_embeddings
import pickle

cfg = PreprocessConfig()
word_embeddings = load_embeddings(cfg)
keys = [key for key,value in tqdm(word_embeddings)]
with open("data/magnitude_keys","wb") as f:
    pickle.dump(keys,f)