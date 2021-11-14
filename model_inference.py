# !!!! How to use: put a new sog into '/audio_test/' directory and start script

from utils.preprocess_class import PreprocessAudio
from utils.audio_processor import AudioProcessor

import os
from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.optim as optim
from pydub import AudioSegment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using - {device}')

resnet_model = resnet34(pretrained=True)
resnet_model.fc = nn.Linear(512, 2)
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet_model = resnet_model.to(device)

####################
### Load model checkpoint/weights/state
####################

model = resnet_model
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load("model_large.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
# print(model)

model.eval()

####################
### Set directories
####################

DATA_FOLDER = '/audio_test/'

####################
### Preprocess new song
####################

preprocessor = PreprocessAudio(open_folder=DATA_FOLDER, save_folder=DATA_FOLDER+'normalized', num_channels=1, frame_rate=22050, audio_format='mp3', chunk_length=5000, chunks_save_folder=DATA_FOLDER+'5sec_chunks')

preprocessor.preprocess()
preprocessor.make_chunks()
preprocessor.display_stats()

####################
### New song choir analysis
####################

predictions = []
combined = []

for idx, fname in enumerate(sorted(os.listdir(f"{DATA_FOLDER}/5sec_chunks"))):
    fname = f'{DATA_FOLDER}/5sec_chunks/' + fname
    aud = AudioProcessor.load_audio(fname)
    sgram = AudioProcessor.generate_mel_spectrogram(aud, n_mels=128, n_fft=1024, hop_len=None)

    input = sgram.unsqueeze(0)

    output = model(input)

    prediction = int(torch.max(output.data, 1)[1].numpy())

    if prediction == 1:
        sound = AudioSegment.from_file(fname, format="mp3")
        combined.append(sound)

    predictions.append(prediction)

file_handle = sum(combined).export(f'{DATA_FOLDER}/combined_choir.mp3', format="mp3")



