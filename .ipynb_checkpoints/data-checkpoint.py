import torch
import torchaudio 
import torchvision
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from tqdm import tqdm 
import os 

file_names = [e for e in os.walk('data')][0][2]
dir_name = 'data'
n_fft = 1024

data = []
for file in tqdm(file_names[:30]):
    waveform, sample_rate = torchaudio.load(f"{dir_name}/{file}")
    data.append(waveform.mean(dim=0, keepdim=True))

SAMPLE_RATE = sample_rate
transform = Spectrogram(n_fft=n_fft, power=None)
inv_transform = InverseSpectrogram(n_fft=n_fft)

X = [] #actual data 
seq_len = 100
for e in tqdm(data):
    e = transform(e)[0]
    for i in range(0, e.shape[1], seq_len):
        X.append(e[:, i:i+seq_len])

