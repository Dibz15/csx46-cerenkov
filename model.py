import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#Neural net model based off of SRCNN seen here: http://personal.ie.cuhk.edu.hk/%7Eccloy/files/eccv_2014_deepresolution.pdf

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

		# define your network here
        #SRCNN defines c = 3 (num input channels), f_1 = 9 (kernel size), and n_1 = 64 (layer output channels)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=k_1, padding=P_1)
        #SRCNN defines f_2 = 1 (kernel size) and n_2 = 32 (layer output channels)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=k_2, padding=P_2)
        #SRCNN defines f_3 = 5 (kernel size). Output color image means n_3 = 3
        self.conv3 = nn.Conv2d(32, 1, kernel_size=k_3, padding=P_3)

        self._initialize_weights()

    def forward(self, x):
        # define your forward pass here
        #Note, we pass each convolution output through ReLU as described by SRCNN
        x = F.relu(self.conv1(x))
        #print("Size 1: ", x.size())
        x = F.relu(self.conv2(x))
        #print("Size 2: ", x.size())
        x = self.conv3(x)

        return x

    def _initialize_weights(self):
        # initialize your weights here
        # Someone on the Internet says Xavier Univorm initialization is a good one
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)
        init.xavier_uniform_(self.conv3.weight)


class Feedforward(nn.Module):
    def __init__(self, input_size):
        super(Feedforward, self).__init__()

        self.input_size = input_size

        self.fc1 = torch.nn.Linear(self.input_size, self.input_size // 2)
        self.fc2 = torch.nn.Linear(self.input_size // 2, self.input_size // 4)
        self.fc3 = torch.nn.Linear(self.input_size // 4, self.input_size // 5)
        self.fc4 = torch.nn.Linear(self.input_size // 5, self.input_size // 5)
        self.fc5 = torch.nn.Linear(self.input_size // 5, self.input_size // 5)
        self.fc6 = torch.nn.Linear(self.input_size // 5, self.input_size // 5)
        self.fc7 = torch.nn.Linear(self.input_size // 5, self.input_size // 10)
        self.fc8 = torch.nn.Linear(self.input_size // 10, self.input_size // 20)
        self.fc9 = torch.nn.Linear(self.input_size // 20, 1)

        self._initialize_weights()

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = torch.sigmoid(self.fc9(x))

        return x

    def _initialize_weights(self):
        pass