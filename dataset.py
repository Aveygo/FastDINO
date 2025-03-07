import os, glob, random, numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from audio import log_mel_spectrogram
import cv2

class MusicDataset(Dataset):
    def __init__(self, mel_dir_src: str):
        self.files = glob.glob(os.path.join(mel_dir_src, "*.npy"))
        
        self.global_size = 6000                         # ~30 seconds
        self.local_size = self.global_size // 4         # ~10 seconds

        self.cache = []

    def __len__(self):
        return len(self.files)
    
    def global_aug(self, data):
        start = random.randint(0, data.shape[-1] - self.global_size - 1)
        return data[:, start: start+self.global_size]
    
    def local_aug(self, data):
        start = random.randint(0, data.shape[-1] - self.local_size - 1)
        return data[:, start: start+self.local_size]

    def __getitem__(self, idx):
        
        if len(self.cache) == 16:
            idx = random.randint(0, 15)
            mel = self.cache[idx]
            if random.randint(0, 16) == 0:
                del self.cache[idx]
        else:
            file_pth = self.files[idx]
            mel = np.load(file_pth)
            if mel.shape[-1] < self.global_size:
                if idx == len(self) - 1:
                    return self[0]
                return self[idx + 1]

            self.cache.append(mel)
        
        x1 = self.global_aug(mel)
        return x1
        x2 = self.local_aug(mel)
        #print(mel.shape)
        
        if random.randint(0, 1):
            noise_amount = np.random.normal() * 0.05            
            x1 = x1 + np.random.normal(size=x1.shape) * noise_amount
            
        if random.randint(0, 1):
            noise_amount = np.random.normal() * 0.05            
            x2 = x2 + np.random.normal(size=x2.shape) * noise_amount  
        
        
        return x1, x2
    

if __name__ == "__main__":
    d = MusicDataset("/home/greg/Datasets/music/mel_features")
    sample = d[1]
    print(sample[0].shape, sample[1].shape)