import math
import optunity
import random
import numpy as np
from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

def convert(pars):
  MAX_CONV_OUTPUT = 24
  MAX_FC_OUTPUT = 1020

  c2 = int(pars['c2'] * MAX_CONV_OUTPUT) + 8
  l1 = int(pars['l1'] * MAX_FC_OUTPUT) + 4
  l2 = int(pars['l2'] * MAX_FC_OUTPUT) + 4

  batch_size = None
  if 0 <= pars['batch_size'] <= 0.25:
    batch_size = 2
  elif 0.25 <= pars['batch_size'] <= 0.50:
    batch_size = 4
  elif 0.50 <= pars['batch_size'] <= 0.75:
    batch_size = 8
  else:
    batch_size = 16

  return {'c2':c2, 'l1':l1, 'l2':l2, 'batch_size':batch_size, 'lr':pars['lr']}



def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset

class Net(nn.Module):
    def __init__(self, c2=64, l1=120, l2=84):
        super(Net, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(32*32*c2, l1),
            nn.ReLU(inplace=True),
            nn.Linear(l1, l2),
            nn.ReLU(inplace=True),
            nn.Linear(l2, 10)
        )

    def forward(self, x_input):
        x_input = self.conv_layer(x_input)
        x_input = x_input.view(x_input.size(0), -1)
        x_input = self.fc_layer(x_input)
        return x_input

def train_cifar(config, num_epochs=3):
  net = Net(config["c2"], config["l1"], config["l2"])
  data_dir = os.path.abspath("./data")

  device = "cpu"
  if torch.cuda.is_available():
      device = "cuda:0"
      if torch.cuda.device_count() > 1:
          net = nn.DataParallel(net)
  net.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

  trainset, _ = load_data(data_dir)
  
  # 80 / 20 train, validation split (from the train dataset)
  test_abs = int(len(trainset) * 0.8)
  train_subset, val_subset = random_split(
      trainset, [test_abs, len(trainset) - test_abs])
  
  trainloader = torch.utils.data.DataLoader(
      train_subset,
      batch_size=int(config["batch_size"]),
      shuffle=True,
      num_workers=8)
  valloader = torch.utils.data.DataLoader(
      val_subset,
      batch_size=int(config["batch_size"]),
      shuffle=True,
      num_workers=8)
  
  for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
    
  # Validation loss
  val_loss = 0.0
  val_steps = 0
  total = 0
  correct = 0
  for i, data in enumerate(valloader, 0):
      with torch.no_grad():
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)

          outputs = net(inputs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

          loss = criterion(outputs, labels)
          val_loss += loss.cpu().numpy()
          val_steps += 1

  return val_loss, net

def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def obj_func(c2, l1, l2, batch_size, lr):
  # Convert:
  # c2 --> int
  # l1 --> int
  # l2 --> int
  # batch_size --> 4 possible ints.
  # lr.

  # return validation set loss.

  pars = {'c2':c2, 'l1':l1, 'l2':l2, 'batch_size':batch_size, 'lr':lr}
  config = convert(pars)
  loss, _ = train_cifar(config)
  return loss

def run_trial(num_samples=50, gpus_per_trial=1):
  data_dir = os.path.abspath("./data")
  load_data(data_dir)

  pars, details, _ = optunity.minimize(
      obj_func, 
      num_evals=num_samples, 
      c2=[0, 1],
      l1=[0, 1],
      l2=[0, 1],
      batch_size=[0, 1],
      lr=[1e-4, 1e-1],
      solver_name='nelder-mead',
  )

  ## Evaluate results.
  print("Best trial config: {}".format(convert(pars)))
  print(f"Best trial final validation loss: {details.optimum}")

  _, best_trained_model = train_cifar(convert(pars))

  device = "cpu"
  if torch.cuda.is_available():
      device = "cuda:0"
      if gpus_per_trial > 1:
          best_trained_model = nn.DataParallel(best_trained_model)
  best_trained_model.to(device)

  test_acc = test_accuracy(best_trained_model, device)
  print("Best trial test set accuracy: {}".format(test_acc))
  return test_acc, convert(pars), details

def main():
  results = []
  for trial in range(50):
      test_acc, best_config, details = run_trial(num_samples=50)
      print(details)
      print(f"Trial #{trial + 1}: ")
      print(f"{test_acc},{best_config}")
      results.append((test_acc, best_config, details))
      with open("nelderMead.txt", "a+") as f:
            f.write(f"{test_acc},{best_config}")
            f.write("\n")

  for r in results:
    print(r)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main()
