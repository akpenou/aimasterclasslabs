import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math

import ipdb;

# import ipdb;
# ipdb.set_trace() 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() # size = [1, 28, 28]
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # [10, 12, 12]
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # [20, 4, 4]
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 27)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
