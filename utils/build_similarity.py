import numpy as np
import json, glob, os
from numpy import dot
from numpy.linalg import norm

features = glob.glob(os.path.join("/home/greg/Datasets/music/mel_features", "*.npy"))

with open('data.json') as f:
    hash_key = json.load(f)

with open('names.json') as f:
    name_key = json.load(f)

embs = np.load("embeddings.npy")

similarity = {}

for feature_pth in features:
    file_hash = feature_pth.split("/")[-1].split(".")[0]
    emb_name = hash_key.get(file_hash.split("/")[-1])
    if emb_name is None:
        continue
    
    if not emb_name in name_key:
        continue
    
    target_idx = int(name_key.index(emb_name))
    target_emb = embs[target_idx]
    distances = np.linalg.norm(embs - target_emb, axis=0)
    #distances = dot(embs, target_emb)/(norm(embs)*norm(target_emb))
    
    closest_idxs = np.argpartition(distances, 5)[:5]
    
    #closest_names = [name_key[i] for i in closest_idxs]
    similarity[target_idx] = [int(i) for i in list(closest_idxs)]
    
with open('similarity.json', 'w') as f:
    json.dump(similarity, f)
