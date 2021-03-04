import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

from utility import load_transform_data
from model import save_checkpoint, train_model, validate_model, test_model


parser = argparse.ArgumentParser(description='Train a neural network on a dataset of images and save the model checkpoint')

# Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
parser.add_argument('--data_dir', type=str, default='flowers',
                        help='set path of the training image folder (default=flowers)')

parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                        help='set path of the model checkpoint (default=checkpoint.pth)')

parser.add_argument('--arch', type=str, default='vgg16',
                    help='Choose cnn model architecture either vgg16 or densenet121 (default=vgg16')

parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='set learning rate (default=0.003')

parser.add_argument('--hidden_units', type=int, default=500,
                        help='set number of hidden units (default=500')

parser.add_argument('--dropout', type=float, default=0.03,
                        help='set dropout rate (default=0.03')

parser.add_argument('--epochs', type=int, default=3,
                        help='set number of epochs (default=3')

parser.add_argument('--device', type=str, default='cuda',
                        help='set device mode cuda or cpu (default=cuda)')

# Retrieving command line argument options
results = parser.parse_args()

data_dir = results.data_dir
save_dir = results.save_dir
arch = results.arch
learning_rate = results.learning_rate
hidden_units = results.hidden_units
drop_out_probability = results.dropout
num_epochs = results.epochs
device = results.device

train_data, test_data, valid_data, trainloader, testloader, validloader = load_transform_data(data_dir)

# Model architecture: The training script allows users to choose from
# least two different architectures available from torchvision.models
# Model hyperparameters: The training script allows users to set hyperparameters for learning rate, number of hidden units
# A new feedforward network is defined for use as a classifier using the features as input
if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(drop_out_probability)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
elif arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(drop_out_probability)),
                          ('fc2', nn.Linear(4096, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(drop_out_probability)),
                          ('fc3', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = classifier

# Define the loss
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

model, optimizer = train_model(model, trainloader, validloader, criterion, num_epochs, optimizer, device)
test_model(model, testloader, device)

save_checkpoint(model, optimizer, train_data, arch, save_dir)
