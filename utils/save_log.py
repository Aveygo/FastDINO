from torch.utils.tensorboard import SummaryWriter
import numpy as np, json

embs = np.load("embeddings.npy")
with open('names.json') as f:
    name_key = json.load(f)
    
writer = SummaryWriter("../logs/log")
writer.add_embedding(
    embs,
    metadata=name_key,
    tag="embeddings",
)
writer.close()