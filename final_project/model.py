import torch
from torch import nn
import mlflow



    

    

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
        
        mlflow.log_param("model_batch_size", self.batch_size)
        mlflow.log_param("model_learning_rate", self.learning_rate)
        mlflow.log_param("model_input_size", self.input_size)
        mlflow.log_param("model_hidden_size", self.hidden_size)
        mlflow.log_param("model_conditioned_input_size", self.conditioned_input_size)
    
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
    
    

 