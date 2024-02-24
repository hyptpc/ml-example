#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

import numpy as np
import matplotlib.pyplot as plt

#______________________________________________________________________________
class ExampleNN(nn.Module):
  ''' Network definition '''
  def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden1_size)
    self.fc2 = nn.Linear(hidden1_size, hidden2_size)
    self.fc3 = nn.Linear(hidden2_size, output_size)

  def forward(self, x):
    z1 = F.relu(self.fc1(x))
    z2 = F.relu(self.fc2(z1))
    return self.fc3(z2)

#______________________________________________________________________________
if __name__ == '__main__':
  device = 'cpu'

  ''' download data '''
  train_data = MNIST(root='data', train=True, download=False,
                     transform=ToTensor())
  test_data = MNIST(root='data', train=False, download=False,
                    transform=ToTensor())
  print(train_data)
  print(test_data)

  ''' size of each layer '''
  input_size = 28*28
  hidden1_size = 1024
  hidden2_size = 512
  output_size = 10

  model = ExampleNN(
    input_size, hidden1_size, hidden2_size, output_size).to(device)

  ''' load model '''
  model.load_state_dict(torch.load('model_digits.pth'))

  ''' test '''
  for i in range(10):
    image, label = test_data[i]
    image = image.view(-1, 28*28).to(device)
    prediction_label = torch.argmax(model(image))
    ax = plt.subplot(1, 10, i+1)
    plt.imshow(image.detach().to('cpu').numpy().reshape(28, 28), cmap='gray')
    ax.axis('off')
    ax.set_title(f'pred {prediction_label}\ncor {label}', fontsize=10)
  plt.show()
