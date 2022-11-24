from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import matplotlib
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_notebook
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



    

    

class CVAE(nn.Module):
    
    batch_size: int = 128
    learning_rate: float = 0.005
    input_size: int
    hidden_size: int
    conditioned_input_size: int

    def __init__(self, input_size: int, conditioned_input_size: int):
        super(CVAE, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = 40
        self.conditioned_input_size = conditioned_input_size
        
        self.fc1 = nn.Linear(self.input_size+self.conditioned_input_size, 80)
        self.fc21 = nn.Linear(80, self.hidden_size)
        self.fc22 = nn.Linear(80, self.hidden_size)
        
        self.relu = nn.ReLU()
        
        self.fc3 = nn.Linear(self.hidden_size, 80)
        self.fc4 = nn.Linear(80, self.input_size)
    
    def encode(self, x, labels):
        x = torch.cat((x, labels), 1)
        x = self.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)
        
    def decode(self, z, labels):
        torch.cat((z, labels), 1)
        z = self.relu(self.fc3(z))
        #return torch.sigmoid(self.fc4(z))
        return self.fc4(z)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 *logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
        
    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z, labels)
        return x
    
    

 