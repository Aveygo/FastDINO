dataset = "/home/greg/Datasets/music/mel_features"
import sys 
sys.path.insert(0, "..")

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import Whisper
from main import DINO1D
#from test import DINO
import json, torch, glob, os, random, numpy as np, time
torch.set_printoptions(precision=4, sci_mode=False)
with open('data.json') as f:
    d = json.load(f)


#dino = DINO1D(Whisper(), Whisper(), "cuda")
dino = DINO1D(Whisper(), "cuda").eval()

dino.load_state_dict(torch.load("/home/greg/Models/fastdino.pth")["model"])


features = glob.glob(os.path.join("/home/greg/Datasets/music/mel_features", "*.npy"))
embs = None
meta = []

global_size = 6000

for feature_pth in features:#[:1000]:
    music_embedding = np.load(feature_pth)
    if music_embedding.shape[-1] < global_size:
        continue
    
    """
    emb_sum = torch.zeros((1, 640))
    for i in range(8):
    
        start = random.randint(0, music_embedding.shape[-1] - global_size - 1)
        data = music_embedding[:, start: start+global_size]
        
        data = torch.from_numpy(data)
        data = data.unsqueeze(0).float().cuda()
        
        emb_sum += dino.primary.encoder(data).mean(-1).detach().cpu()
    
    emb = emb_sum / 8
    """
    file_hash = feature_pth.split("/")[-1].split(".")[0]
    emb_name = d.get(file_hash.split("/")[-1])
    if emb_name is None:
        continue
    
    start = random.randint(0, music_embedding.shape[-1] - global_size - 1)
    data = music_embedding[:, start: start+global_size]
    data = torch.from_numpy(data).float().cuda().unsqueeze(0)
    
    with torch.no_grad():
        emb = dino.primary.encoder(data)
    emb=emb.mean(-1).detach().cpu()
    
    #print(emb)
    
    if embs is None:
        embs = emb
    else:
        embs = torch.concat([embs, emb], axis=0)
    
    author = emb_name.split("/")[-3]
    meta.append(emb_name)
    print(file_hash)
    
print(embs.shape)

embs = embs.numpy()
np.save("embeddings.npy", embs)

import json
with open('names.json', 'w') as f:
    json.dump(meta, f)



#writer = SummaryWriter("logs/log")
#writer.add_embedding(
#    embs,
#    metadata=meta,
#    tag="embeddings",
#)
#writer.close()
"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from matplotlib import cm

embeddings_np = embs.detach().cpu().numpy()

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_np)

label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(meta)
colormap = plt.colormaps.get_cmap("plasma")

colors = colormap(numeric_labels / len(np.unique(numeric_labels)))

# Step 3: Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=5)
plt.colorbar(scatter, label='Class Labels')
plt.title('t-SNE visualization of embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig("main.png")"""

