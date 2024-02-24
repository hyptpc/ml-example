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
def train_model(model, train_loader, criterion, optimizer, device='cpu'):
  ''' training function '''
  model.train() # train mode
  for i, (images, labels) in enumerate(train_loader):
    images, labels = images.view(-1, 28*28).to(device), labels.to(device)
    optimizer.zero_grad() # initialize grad
    #1 forward
    outputs = model(images)
    #2 calculate loss
    loss = criterion(outputs, labels)
    #3 calculate grad
    loss.backward()
    #4 update parameters
    optimizer.step()
  return loss.item()

#______________________________________________________________________________
def test_model(model, test_loader, criterion, optimizer, device='cpu'):
  ''' test function '''
  model.eval() # eval mode
  with torch.no_grad(): # invalidate grad
    for i, (images, labels) in enumerate(test_loader):
      images, lebels = images.view(-1, 28*28).to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)
  return loss.item()

#______________________________________________________________________________
def learning(model, train_loader, test_loader, criterion,
             optimizer, n_epoch, device='cpu'):
  ''' leaning function '''
  train_loss_list = []
  test_loss_list = []
  # epoch loop
  for epoch in range(1, n_epoch+1, 1):
    train_loss = train_model(model, train_loader, criterion, optimizer,
                             device=device)
    test_loss = test_model(model, test_loader, criterion, optimizer,
                           device=device)
    print(f'epoch : {epoch}, '+
          f'train_loss : {train_loss:.5f}, test_loss : {test_loss:.5f}')
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
  return train_loss_list, test_loss_list

#______________________________________________________________________________
if __name__ == '__main__':

  ''' download data '''
  train_data = MNIST(root='data', train=True, download=True,
                     transform=ToTensor())
  test_data = MNIST(root='data', train=False, download=True,
                    transform=ToTensor())
  print(train_data)
  print(test_data)

  ''' plot sample digit '''
  plt.imshow(train_data[0][0].squeeze(), cmap='gray')
  plt.colorbar()
  plt.show()

  ''' set mini-batch data '''
  batch_size = 256
  train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                            shuffle=True)
  test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
                           shuffle=True)

  ''' size of each layer '''
  input_size = 28*28
  hidden1_size = 1024
  hidden2_size = 512
  output_size = 10

  device = 'cpu'
  model = ExampleNN(
    input_size, hidden1_size, hidden2_size, output_size).to(device)
  print(model)

  ''' Loss function '''
  # MSELoss : mean squared error, Sum(x-y)^2
  # CrossEntropyLoss : -Sum(x*log(y))
  # criterion = nn.MSELoss()
  criterion = nn.CrossEntropyLoss()

  ''' Optimizer '''
  # SGD : stochastic gradient descent
  # lr : learning rate
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  # optimizer = optim.Adam(model.parameters(), lr=0.0001)

  ''' learning '''
  n_epoch = 10
  train_loss_list, test_loss_list = learning(
    model, train_loader, test_loader, criterion, optimizer,
    n_epoch, device=device)

  ''' plot loss '''
  plt.plot(range(len(train_loss_list)), train_loss_list,
           c='b', label='train loss')
  plt.plot(range(len(test_loss_list)), test_loss_list,
           c='r', label='test loss')
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.legend()
  plt.grid()
  plt.show()

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

  ''' save model '''
  torch.save(model.state_dict(), 'model_digits.pth')
