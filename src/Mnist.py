import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import copy
import random

from tqdm import tqdm
from pathlib import Path
from numpy import linalg as LA
from src.models import Mnist_CNN
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

class Mnist_SVDD(object):

    def __init__(self, config):
        self.config = config
        self.model = None
        self.center = None

        if config.normal == -1:
            self.outliers = config.outliers
            self.normal = np.arange(0,10)
            self.normal = self.normal[self.normal != self.outliers]
        elif config.outliers == -1:
            self.normal = config.normal
            self.outliers = np.arange(0,10)
            self.outliers = self.outliers[self.outliers != self.normal]

        self.load_datasets()

        if config.load_model:
            self.load_model(config.mnist_cnn_weights)
        else:
            self.model = Mnist_CNN(config)
            self.center = self.get_center()
        self.lambda_regularizer = config.lambda_regularizer / 2.0
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def get_samples(self, dataset, label):
        num_samples = np.random.randint(0, self.config.max_samples)
        idx = dataset.targets==label
        idx = [i for (i, x) in enumerate(idx) if x]
        samples = random.sample(idx, num_samples)
        return samples
    
    def load_datasets(self):
        training_data = datasets.MNIST(
            root="./data/",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.MNIST(
            root="./data/",
            train=False,
            download=True,
            transform=ToTensor()
        )

        # selecting random samples from each outlier class (training set)
        idx = training_data.targets==self.normal
        for i in self.outliers:
            samples = self.get_samples(training_data, i)
            idx[samples] = True
        training_data.targets = training_data.targets[idx]
        training_data.data = training_data.data[idx]

        # selecting random samples from each outlier class (test set)
        idx = test_data.targets==self.normal
        for i in self.outliers:
            samples = self.get_samples(test_data, i)
            idx[samples] = True
        test_data.targets = test_data.targets[idx]
        test_data.data = test_data.data[idx]

        # spliting training set into training/val sets
        a = int(0.85*len(training_data))
        b = len(training_data) - a
        train_subset, val_subset = torch.utils.data.random_split(
            training_data,
            [a, b],
            generator=torch.Generator().manual_seed(1))

        self.train_dataloader = DataLoader(train_subset,
                                           batch_size=self.config.batch_size,
                                           shuffle=True)
        self.val_dataloader = DataLoader(val_subset,
                                         batch_size=self.config.batch_size,
                                         shuffle=False)
        self.test_dataloader = DataLoader(test_data,
                                          batch_size=self.config.batch_size,
                                          shuffle=False)
            
    def load_model(self, path):
        if Path(path).is_file():
            print("Loading existing weights...")
            state_dict = torch.load(path)
            model = Mnist_CNN(self.config)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()
            center = state_dict['center']
            self.model = model
            self.center = center
        else:
            raise Exception("The provided path does not contain weights")

    def save_model(self, path, model_state_dict=None):
        if model_state_dict is None:
            model_state_dict = self.model.state_dict()
        torch.save({'model_state_dict':model_state_dict,
                    'center':self.center}, path)

    def loss_func(self, x):
        return torch.cdist(x, self.center.view(1, -1)).mean()

    def get_regularizer_term(self):
        l2_lambda = self.lambda_regularizer
        l2_reg = torch.tensor(0.)
        i = 0
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
            i+=1
        return l2_lambda*(l2_reg/i)

    def loss_batch(self, xb):
        loss = self.loss_func(self.model(xb))
        loss += self.get_regularizer_term()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        return loss

    def get_center(self):
        print("Computing center...")
        with torch.no_grad():
            center = torch.zeros(self.config.output_channels)
            for i, (xb, _) in enumerate(tqdm(self.train_dataloader)):
                features = self.model(xb)
                center += torch.mean(features, 0)
                if i>0:
                    center /= 2.
        return center

    def fit(self):
        """
        wandb.init(project='DEEP-SVDD')
        wandb_config = wandb.config
        wandb_config.learning_rate = self.config.lr
        wandb.watch(self.model)
        """
        
        print("Training model...")
        center = self.center
        model = self.model
        n_epochs = self.config.epochs
        best_val_loss = np.inf
        for epoch in range(n_epochs):
            model.train()
            loop = tqdm(self.train_dataloader)
            for xb, _ in loop:
                loss = self.loss_batch(xb)
                loop.set_description("Epoch [{}/{}] ".format(epoch, n_epochs))
                loop.set_postfix({"loss":loss.item()})

            model.eval()
            with torch.no_grad():
                losses = [torch.cdist(model(xb), center.view(1, -1))
                          for xb, yb in self.val_dataloader]
                losses = [x.item() for xb in losses for x in xb]
                val_loss = np.mean(losses) + self.get_regularizer_term()
                print("val_loss={:.6f}".format(val_loss))

            if val_loss < best_val_loss:
                best_model_state = copy.deepcopy(model.state_dict())
                best_val_loss = val_loss
                self.save_model(self.config.mnist_cnn_weights, best_model_state)

    def test(self):
        center = self.center
        model = self.model
        model.eval()
        with torch.no_grad():
            img_and_dist = []
            for xb, yb in self.test_dataloader:
                for x in xb:
                    img_and_dist.append((x,torch.cdist(model(x), center.view(1, -1)).item()))
            sorted_by_biggest_dist = sorted(img_and_dist,
                                            key = lambda tup: tup[1],
                                            reverse=True)
            ten_furthest_from_center = sorted_by_biggest_dist[:10]
            
            sorted_by_shortest_dist = sorted(img_and_dist,
                                             key = lambda tup: tup[1])
            ten_closest_to_center = sorted_by_shortest_dist[:10]
                    
            losses = [torch.cdist(model(xb), center.view(1, -1))
                      for xb, yb in self.test_dataloader]
            losses = [x.item() for xb in losses for x in xb]
            test_loss = np.mean(losses) + self.get_regularizer_term()
            print("test_loss={:.6f}".format(test_loss))

            fig, axs = plt.subplots(2, 10)
            for i in range(10):
                axs[0, i].imshow(ten_furthest_from_center[i][0].view(28,28))
                axs[1, i].imshow(ten_closest_to_center[i][0].view(28,28))
            plt.show()
            
        
