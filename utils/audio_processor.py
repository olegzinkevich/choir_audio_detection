import torchaudio
from torchaudio import transforms

class AudioProcessor():
    '''Performs audio processing, creates mel spectrogram and tensor'''

    @staticmethod
    def load_audio(audio_file):
        '''
        Loads an audio file as a tensor.
        Returns tensor and sample rate.
        '''
        aud_tensor, sample_rate = torchaudio.load(audio_file)
        return (aud_tensor, sample_rate)

    @staticmethod
    def generate_mel_spectrogram(aud, n_mels=128, n_fft=1024, hop_len=None):
        '''
        Generates Mel Spectrogram from an audio tensor
        Parameters:
        ---------------
        n_mels: int
            Number of mel filterbanks
        n_fft: int
            Number of bins
        hop_len: int
            Len of hops between windows
        '''
        aud_tensor, sample_rate = aud

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spectrogram = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(aud_tensor)

        return (spectrogram)