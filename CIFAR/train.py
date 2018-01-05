"""
http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import torch
import torchvision #Built in for cv stuff
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# General steps
# 1 Loading and normalizing data
# 2 Define ConvNet
# 3 Define a loss function
# 4 Train the network on training data
# 5 Test the network on test data


# Compose chains transforms together
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the data
trainset = torchvision.datasets.CIFAR10(root='./data', train = True,
                                        download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
                                          shuffle = True, num_workers = 2)


testset = torchvision.datasets.CIFAR10(root='./data', train = False,
                                       download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4,
                                         shuffle = False, num_workers =2)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img /2 + 0.5 # unnormalize the image
    npimg = img.numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))

# print labels
print(" ".join('%5x' % classes[labels[j]] for j in range(4)))




