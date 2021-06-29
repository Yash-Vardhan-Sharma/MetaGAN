import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
import random
from tensorboardX import SummaryWriter
import torch.optim as optim



class dataset(torch.utils.data.Dataset):    
    def __init__(self , train=True):
        self.data= pd.read_csv("/home/sever2users/Desktop/MetaGAN/datasets/chess.csv" , sep = ";" ).to_numpy()

        if train :
          self.data = self.data[self.data[:,-1]<12]
        else :
          self.data = self.data[self.data[:,-1]>=12]


    def __getitem__(self, index):
        x, y = torch.tensor((self.data[index,:-1] - np.min(self.data[index,:-1] , axis=0)) / ((np.max(self.data[index,:-1], axis=0)) - np.min(self.data[index,:-1], axis=0)) ), torch.tensor(self.data[index,-1])

        return x,y

    def __len__(self):
        return len(self.data)


class Discriminator(nn.Module):
    
    def __init__(self,ways):
        super(Discriminator,self).__init__()  
        self.fc1 = nn.Linear(23, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,ways)
        self.relu = nn.ReLU()
        self.activation = nn.Softmax(dim=1)

    def forward(self,x):


        x = self.fc3(  self.relu(self.fc2(self.relu(self.fc1(x))))  )        
        D_output = self.activation(  x  )

        return D_output


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner ,loss ,adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):

        disc_logit = learner(adaptation_data.type(torch.cuda.FloatTensor))

 
        task_train_error = loss(disc_logit.reshape(ways*shots , ways) , evaluation_labels.reshape(ways*shots) )

        train_error = task_train_error

        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data.type(torch.cuda.FloatTensor))
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)

    return valid_error, valid_accuracy

def test( fast_lr ,
          loss,
          adaptation_steps,
          shots,
          ways,
          device,
          maml,
          valid_tasks):

    test_size = 200

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(test_size):

        # Compute meta-testing loss
        learner = maml.clone()
        batch = valid_tasks.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                          loss,
                                                          adaptation_steps,
                                                          shots,
                                                          ways,
                                                          device)
        
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()

    print('\n')
    print('Meta Test Error', meta_test_error / test_size)
    print('Meta Test Accuracy', meta_test_accuracy / test_size)

    return  meta_test_error / test_size , meta_test_accuracy / test_size



def main(
        ways=4,
        shots=5,
        meta_lr=0.01,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=10000,
        cuda=True,
        seed=42,
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Create dataset

    train_transforms = [
        NWays(train_dataset, ways),
        KShots(train_dataset, shots*2),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset)
    ]

    valid_transforms = [
            NWays(valid_dataset, ways),
            KShots(valid_dataset, shots*2),
            LoadData(valid_dataset),
            RemapLabels(valid_dataset),
            ConsecutiveLabels(valid_dataset)
        ]


    train_tasks = l2l.data.TaskDataset(train_dataset,
                                          task_transforms=train_transforms,
                                          num_tasks=1000)

    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                          task_transforms=valid_transforms,
                                          num_tasks=200)

    # Create model
    model = Discriminator(ways=ways)
    
    model.to(device)
    
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr )
    
    loss = nn.CrossEntropyLoss(reduction='mean')
    
    best_acc = 0.0

    writer = SummaryWriter("/home/sever2users/Desktop/MetaGAN/maml/viz/chess")

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0

        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = valid_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        if iteration%500 == 0 :
	        print('\n')
	        print('Iteration', iteration)
	        print('Meta Train Error', meta_train_error / meta_batch_size)
	        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
	        print('Meta Valid Error', meta_valid_error / meta_batch_size)
	        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        writer.add_scalar("{}way_{}shot_{}train_as/accuracy/valid".format(ways , shots,adaptation_steps),  (meta_valid_accuracy / meta_batch_size) , iteration)
        writer.add_scalar("{}way_{}shot_{}train_as/accuracy/train".format(ways , shots,adaptation_steps),  (meta_train_accuracy / meta_batch_size) , iteration)

        writer.add_scalar("way_{}shot_{}train_as/loss/valid".format(ways, shots,adaptation_steps),  (meta_valid_error / meta_batch_size) , iteration)
        writer.add_scalar("way_{}shot_{}train_as/loss/train".format(ways, shots,adaptation_steps),  (meta_train_error / meta_batch_size) , iteration)

        

        if ( (meta_valid_accuracy / meta_batch_size) > best_acc ) :

          maml_test = maml.clone()
          
          best_acc = (meta_valid_accuracy / meta_batch_size)   

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    print("\n")
    print(ways,"WAYS",shots,"SHOTS",adaptation_steps,"train_adaptation_steps")
    
    x = []
    
    for a_s in [1,3,10,20]:
      for fast_lr in [0.5 , 0.1 , 0.05 , 0.01] :

        l , a = test(fast_lr,
                        loss,
                         a_s,
                       shots,
                        ways,
                      device,
                   maml_test,
                  valid_tasks)
        
        x.append([a_s , fast_lr , a])
        
        print("test_a_s=",a_s,'fast_lr',fast_lr,"acc=",a)
    
    np.savetxt( '/home/sever2users/Desktop/MetaGAN/maml/test_results/chess/{}way_{}shot_{}train_adp_steps.txt'.format(ways,shots,adaptation_steps), np.array(x))

    torch.save(maml_test.module.state_dict(), '/home/sever2users/Desktop/MetaGAN/maml/saved_models/chess/state_dict_{}way_{}shot_{}train_adp_steps.pth'.format(ways,shots,adaptation_steps) )  


if __name__ == '__main__':

    

    train_dataset = dataset(train=True)
    valid_dataset = dataset(train=False)

    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    
    ways= [4]
    shots= [5]
    meta_lr=0.01
    fast_lr=0.5
    meta_batch_size=32
    adaptation_steps=[1]
    num_iterations=8000
    
    for way in ways:

      for shot in shots :
        
        for a_s in adaptation_steps :
            
            main(ways=way,
                shots=shot,
                meta_lr=meta_lr,
                fast_lr=fast_lr,
                meta_batch_size=meta_batch_size,
                adaptation_steps=a_s,
                num_iterations=num_iterations,
                cuda=True,
                seed=42)   
