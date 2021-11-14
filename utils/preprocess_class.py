from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import traceback

class PreprocessAudio():

    def __init__(self, open_folder, save_folder, chunks_save_folder, num_channels, frame_rate, audio_format, chunk_length):
        '''
        Parameters:
        ---------------
            open_folder, save_folder: str
                folders in which perform audio normalizing
            chunks_save_folder: str
                folder to save audio chunks after normalizing
            num_channels: int
                1 for mono, 2 for stereo
            frame_rate: int
                In hertz - default value is 22050
            audio_format: str
                mp3 or wav
            chunk_length: int
                audio chunk in milliseconds (5000 = 5 sec)
        '''
        self.open_folder = open_folder
        self.save_folder = save_folder
        self.chunks_save_folder = chunks_save_folder
        self.num_channels = num_channels
        self.frame_rate = frame_rate
        self.audio_format = audio_format
        self.chunk_length = chunk_length
        self._stats = 0

    def preprocess(self):
        """The method performs audio normalizing:
        - converts stereo to mono or vice versa,
        - changes sample rate to the specified rate
        """

        dir_ls = [_ for _ in os.listdir(self.open_folder) if _.endswith(self.audio_format)]

        if os.path.isdir(self.save_folder):
            pass
        else:
            os.mkdir(self.save_folder)

        for fname in dir_ls:

            try:
                print('processing file:', fname)
                sound = AudioSegment.from_file(f"{self.open_folder}/{fname}")

                sound = sound.set_channels(self.num_channels)
                sound = sound.set_frame_rate(self.frame_rate)

                sound.export(f"{self.save_folder}/normalized_{fname}", format=self.audio_format)
                self._stats += 1

            except:
                print(traceback.format_exc())


    def make_chunks(self):
        """The method performs audio chunking:
        - devides a single audio in the 5 seconds fragments and saves them as files
        """

        if os.path.isdir(self.chunks_save_folder):
            pass
        else:
            os.mkdir(self.chunks_save_folder)

        dir_ls = [_ for _ in os.listdir(self.save_folder) if _.endswith(self.audio_format)]

        for fname in dir_ls:
            print('processing:', fname)

            try:
                sound = AudioSegment.from_file(f"{self.save_folder}/{fname}" , 'mp3')
                chunk_length_ms = self.chunk_length

                chunks = make_chunks(sound, chunk_length_ms)

                for i, chunk in enumerate(chunks):
                    chunk_name = f"{self.chunks_save_folder}/{i}_{fname}".format(i)
                    chunk.export(chunk_name, format=self.audio_format)

            except:
                print(traceback.format_exc())


    def display_stats(self):

        print()
        print(f'processed -- {self._stats}.{self.audio_format} -- files')


if __name__ == '__main__':

    preprocessor = PreprocessAudio(open_folder='C:/dev/datasets/chorus/not_chorus/disco', save_folder='C:/dev/datasets/chorus/not_chorus/disco/normalized', num_channels=1, frame_rate=22050, audio_format='wav', chunk_length=5000, chunks_save_folder='C:/dev/datasets/chorus/not_chorus/disco/5sec_chunks')

    preprocessor.preprocess()
    preprocessor.make_chunks()
    preprocessor.display_stats()

