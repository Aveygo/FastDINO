import os, glob, hashlib, numpy as np, tqdm, random, json
from audio import log_mel_spectrogram, load_audio

MAX_BYTES_SIZE = 1e+7 # 10mb
DST = "/home/greg/Datasets/music/mel_features"

music_dir_src = "/home/greg/Documents/Music/Music/"
pths = glob.glob(os.path.join(music_dir_src, "**"), recursive=True)
pths = [i for i in pths if i.endswith(".mp3")]

random.Random(42).shuffle(pths)

pths = pths[:10000]

data = {}

for idx, music_src in enumerate(tqdm.tqdm(pths)):
    
    #print(f"[{idx}/{len(pths)}] - {(idx/len(pths))*100:.2f}%")
    
    if os.path.getsize(music_src) > MAX_BYTES_SIZE:
        continue
    
    file_hash = hashlib.md5(music_src.split("/")[-1].encode()).hexdigest()
    data[file_hash] = music_src
    file_dst = os.path.join(DST, file_hash + ".npy")
    
    if not os.path.exists(file_dst):
        try:
            mel = log_mel_spectrogram(music_src)
            mel = mel.numpy()
            np.save(file_dst, mel)
        except:
            pass

with open('data.json', 'w') as f:
    json.dump(data, f)