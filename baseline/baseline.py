import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import learn2learn as l2l
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from random import sample

class dataset(torch.utils.data.Dataset):    
    def __init__(self , train=True , labels=None):
        self.data= pd.read_csv("/home/sever2users/Desktop/MetaGAN/datasets/pendigits_csv.csv").to_numpy()


        if train :
          self.data = self.data[self.data[:,-1]<6]
        else :
          self.data = self.data[self.data[:,-1]>=6]
          self.data1= self.data[self.data[:,-1]>=6]

          # if labels is not None:

          #   for i , label in enumerate(labels):

          #     if label == labels[0]:
          #       self.data = self.data[(self.data[:,-1] == label) ]
          #       self.data[:,-1] = 0
              
          #     else :
          #       x= self.data1[(self.data1[:,-1] == label) ]
          #       x[:,-1] = i 
          #       self.data = np.concatenate((self.data ,x) , axis=0 )

          #   iitr


    def __getitem__(self, index):
        x, y = torch.tensor((self.data[index,:-1] - np.min(self.data[index,:-1] , axis=0)) / ((np.max(self.data[index,:-1], axis=0)) - np.min(self.data[index,:-1], axis=0)) ), torch.tensor(self.data[index,-1])

        return x,y

    def __len__(self):
        return len(self.data)

    def unique(self):
        return np.unique( self.data[:,-1])

train_dataset = dataset(train=True)
valid_dataset = dataset(train=False)

device='cuda'
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# baseline = Discriminator(6).to(device)
# base_optim = optim.Adam(baseline.parameters(), 1e-4)

train_loader = torch.utils.data.DataLoader(
                                 train_dataset, 32,
                                 shuffle=True, num_workers=1)

valid_loader = torch.utils.data.DataLoader(
                                 valid_dataset, 32,
                                 shuffle=False, num_workers=1)

def nCr(data,r):
  return np.random.choice(data,r)


class Discriminator(nn.Module):
    
    def __init__(self, odim):
        super(Discriminator,self).__init__()  
        self.fc1 = nn.Linear(16, 24)
        self.fc2 = nn.Linear(24,32)
        self.fc3 = nn.Linear(32,odim)
        self.relu = nn.ReLU()
        self.activation = nn.Softmax(dim=1)

    def forward(self,x):


        x = self.fc3(  self.relu(self.fc2(self.relu(self.fc1(x))))  )
        
        D_output = self.activation(  x  )

        # last_logit = self.activation(x[:,-1].reshape(1,-1))


        return D_output

def train(baseline, train_loader):
  total=0
  correct=0
  total_loss=0

  criterion=nn.CrossEntropyLoss(reduction='mean')

  for iteration in range(500):
      for i, (input, target) in enumerate(train_loader):
          # data=train_tasks.sample()
          inputs,targets=input.type(torch.cuda.FloatTensor), target.to(device)
          base_optim.zero_grad()

          outputs=baseline(inputs.type(torch.cuda.FloatTensor))
          loss=criterion(outputs,targets)
          loss.backward()
          base_optim.step()

          total_loss += loss.item()
          _, predicted = outputs.max(1)
          total += targets.size(0)
          correct += predicted.eq(targets).sum().item()
          accuracy = correct*100. / total

          train_result = {
                        'accuracy': correct*100. / total,
                        'loss': loss.item(),
                    }

      print(iteration,train_result)
      total=0
      correct=0
      total_loss=0

  torch.save(baseline.state_dict(), "/content/baseline_train_state_dict.pth")


def test(path, valid_dataset ,ways):

  sd=torch.load(path)

  task = nCr(valid_dataset.unique(),ways)

  new_valid_dataset = dataset(train=False, labels=task)

  valid_loader = torch.utils.data.DataLoader(
                                 valid_dataset, 4,
                                 shuffle=False, num_workers=1)


  test_baseline=Discriminator(ways)
  tes_sd=test_baseline.state_dict()
  sd['fc3.weight']=tes_sd['fc3.weight']
  sd['fc3.bias']=tes_sd['fc3.bias']

  test_baseline.load_state_dict(sd)
  test_baseline=test_baseline.to(device)

  total=0
  correct=0
  total_loss=0

  base_optim_new = optim.Adam(test_baseline.parameters(), 1e-2)

  criterion=nn.CrossEntropyLoss(reduction='mean')

  for iteration in range(20):
      for i, (input, target) in enumerate(valid_loader):
              
          if i==0:
            input[0] = torch.tensor([0.0000, 0.8500, 0.3800, 1.0000, 0.8100, 0.8800, 0.8700, 0.5000, 0.8400,
                  0.1200, 0.5800, 0.0000, 0.5300, 0.2200, 1.0000, 0.2400],
                 dtype=torch.float64)
            target[0] = 7

          target=target-6

          # data=train_tasks.sample()

          inputs,targets=input.type(torch.cuda.FloatTensor), target.to(device)
          base_optim_new.zero_grad()

          outputs=test_baseline(inputs.type(torch.cuda.FloatTensor))
          loss=criterion(outputs,targets)

          if i == 0:
            loss.backward()
            base_optim_new.step()

          if i != 0:
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = correct*100. / total

            train_result = {
                          'accuracy': correct*100. / total,
                          'loss': loss.item(),
                      }
        
      print(iteration,train_result)
      total=0
      correct=0
      total_loss=0

  return(train_result)  



if __name__== '__main__':

  path='/home/sever2users/Desktop/MetaGAN/baseline/baseline_train_state_dict.pth'
  # train()

  tacc=[]
  tloss=[]
  ways=4
  for i in range(30):
    res=test(path, valid_dataset, ways)
    tacc.append(res['accuracy'])
    tloss.append(res['loss'])

  print(np.array(tacc).mean(), np.array(tacc).std())


