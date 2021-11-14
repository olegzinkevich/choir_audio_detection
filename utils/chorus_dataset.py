from utils.audio_processor import AudioProcessor
import os
from torch.utils.data import Dataset
import torch
import pandas as pd

class ChorusDataset(Dataset):
    '''
    Creates a Dataset instance for pytorch model training.
    Loads audio files with corresponding labels from csv.
    Performes audio to spectrogram and tensor transformation.
    '''

    def __init__(self, root_dir, annotation_file):

        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)

    def __len__(self):

        return len(self.annotations)

    def __getitem__(self, index):

        img_id = self.annotations.iloc[index, 0]
        filename = os.path.join('/Huawei/', img_id)
        aud = AudioProcessor.load_audio(filename)
        sgram = AudioProcessor.generate_mel_spectrogram(aud, n_mels=128, n_fft=1024, hop_len=None)
        y_label = torch.tensor(float(self.annotations.iloc[index, 1]))
        y_label = y_label.type(torch.LongTensor)

        return (sgram, y_label)