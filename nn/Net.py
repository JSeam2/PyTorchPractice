import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

"""
http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 1 input image channel, 6 output channels, 5x5 square conv kernel

        # Convolutional layer
        # Applies a 2D Convolutional layer over an input signal
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        # Fully Connected Layer
        # Linear Applies a linear transformation to the incoming data
        # y = Ax + b 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        # max pool over (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # Note that you only need to specify a single number
        # if the size is square
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # view function is used to reshape the tensor
        # parameter -1 is used when you don't know number of rows
        # but you are sure of the number of columns
        # only one axis value can be -1
        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


    def num_flat_features(self, x):
        # get the sizes of all dimensions except for the batch dimension
        # which is at 0
        size = x.size()[1:]
        num_features = 1

        for s in size:
            num_features *= s

        return num_features


net = Net()
#print(net)

# output:
#Net(
#      (conv1): Conv2d (1, 6, kernel_size=(5, 5), stride=(1, 1))
#      (conv2): Conv2d (6, 16, kernel_size=(5, 5), stride=(1, 1))
#      (fc1): Linear(in_features=400, out_features=120)
#      (fc2): Linear(in_features=120, out_features=84)
#      (fc3): Linear(in_features=84, out_features=10)
#)

#params = list(net.parameters())
## shows all the param
## print(params)
#print(len(params))
#print(params[0]) #conv1 params

input_var = Variable(torch.randn(1,1,32,32))
out = net(input_var)
#print(out)

# NOTE torch.nn only supports inputs that are a mini-batch of samples,
# and not a single sample. If you have a sinlge sample use input.unsqueeze(0)
# to add a fake batch dimension


output = net(input_var)
target = Variable(torch.arange(1, 11))

# More info on loss found here
# http://pytorch.org/docs/master/nn.html#loss-functions
criterion = nn.MSELoss()
loss = criterion(output, target)

#print(loss)

#input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#      -> view -> linear -> relu -> linear -> relu -> linear
#      -> MSELoss
#      -> loss

# NOTE loss.backward() effectively differentiates the graph wrt to the loss
# All variables in the graph will have their grad Variable accumulated
# in the gradient
# print(loss.backward())

# elucidating the steps
#print(loss.grad_fn) #MSELoss
#print(loss.grad_fn.next_functions[0][0]) # linear layers
#print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #ReLU


# Update the weights
#learning_rate = 0.01
#for f in net.parameters():
#    f.data.sub_(F.grad.data * learning_rate)

# for more complicated update stuff there is torch.optim which contains
# most of the useful optimizers
# Example

import torch.optim as optim

# Defining our optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# In training loop
optimizer.zero_grad() # zero the gradient buffers 
output = net(input_var)
loss = criterion(output, target)
loss.backward()
optimizer.step() # does the update





