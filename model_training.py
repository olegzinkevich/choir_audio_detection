from utils.generate_dataset_csv import generate_dataset_csv
from utils.chorus_dataset import ChorusDataset

import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.optim as optim

####################
### Set directories
####################

DATA_FOLDER = '/Huawei'

####################
### create CSV db
####################

generate_dataset_csv(chorus_path=f'{DATA_FOLDER}/train_normalized/chorus', not_chorus_path=f'{DATA_FOLDER}/train_normalized/not_chorus')

####################
### create Dataset
####################

ds = ChorusDataset(os.getcwd(), 'train.csv')
print(ds[2])
print(len(ds))

####################
### Train / Validation split
####################

val_size = int(len(ds)*0.3)
train_size = len(ds)- int(len(ds)*0.3)
train_set, validation_set = random_split(ds, [train_size, val_size])
print(len(train_set), len(validation_set))

####################
### Data loaders
####################

num_epochs = 10
learning_rate = 0.00001
batch_size = 16
shuffle = True
pin_memory = True
num_workers = 1

train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

####################
### Check data loaders
####################

single_batch = next(iter(train_loader))
print(single_batch[0].shape)

####################
### Set device type
####################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using - {device}')

####################
### Resnet model fine-tune
####################

resnet_model = resnet34(pretrained=True)
resnet_model.fc = nn.Linear(512, 2)
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet_model = resnet_model.to(device)

learning_rate = 2e-4
optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate)
epochs = 10
loss_fn = nn.CrossEntropyLoss()

resnet_train_losses = []
resnet_valid_losses = []

####################
### Model train function
####################

def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer,
          train_losses, valid_losses, change_lr=None):

    for epoch in tqdm(range(1, epochs+1)):

        model.train()
        batch_losses = []

        if change_lr:
            optimizer = change_lr(optimizer, epoch)

        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()

        train_losses.append(batch_losses)
        print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')

        model.eval()
        batch_losses=[]
        trace_y = []
        trace_yhat = []

        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())
            batch_losses.append(loss.item())

        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
        print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')


####################
### Perform model training
####################

train(resnet_model, loss_fn, train_loader, validation_loader, epochs, optimizer, resnet_train_losses, resnet_valid_losses)


####################
### Save model checkpoint/weights
####################

PATH = f"{DATA_FOLDER}/model_checkpoint/model_large.pt"

torch.save({
    'epoch': epochs,
    'model_state_dict': resnet_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),

}, PATH)

