import os
import pandas as pd

DATA_FOLDER = ''

def generate_dataset_csv(chorus_path, not_chorus_path, file_number_limit=5000):
    '''Generates csv with the dataset files location and labels.
        Parameters:
        ---------------
        chorus_path: str
            folder with chorus samples
        not_chorus_path: str
            folder with non chorus samples
        limit: int
            how many samples to use for model training
    '''
    data = []

    for idx, fname in enumerate(sorted(os.listdir(chorus_path))):
        if idx > file_number_limit:
            break
        data.append(('train_normalized/chorus/' + fname, 1))

    for idx, fname in enumerate(sorted(os.listdir(not_chorus_path))):
        if idx > file_number_limit:
            break
        data.append(('train_normalized/not_chorus/' + fname, 0))

    train_df = pd.DataFrame(data, columns=['img_name', 'label'])
    print(train_df)
    train_df.to_csv(f'{DATA_FOLDER}/train.csv', index = False, header=True)


if __name__ == '__main__':

    generate_dataset_csv(chorus_path=f'{DATA_FOLDER}/train_normalized/chorus', not_chorus_path=f'{DATA_FOLDER}/train_normalized/not_chorus',)