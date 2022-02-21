from tqdm import tqdm
from config import PreprocessConfig
from utils import load_embeddings
import pickle

cfg = PreprocessConfig()
word_embeddings = load_embeddings(cfg)
# keys = [key for key,value in tqdm(word_embeddings)]
# with open("data/magnitude_keys.pkl","wb") as f:
#     pickle.dump(keys,f)
word2idx=dict() #単語→magnitudeのIDへの変換辞書
for i, (key,value) in tqdm(enumerate(word_embeddings),total=len(word_embeddings)):
    word2idx[key] = i #magnitudeのqueryingが遅いので辞書で持っておく
with open("data/magnitude_word2idx.pkl","wb") as f:
    pickle.dump(word2idx,f)