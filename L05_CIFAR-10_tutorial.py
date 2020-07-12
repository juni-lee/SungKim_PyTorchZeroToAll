import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 == torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x



if __name__ == "__main__":
    """
    1. Loading and normalizaing CIFAR10
    
    The output of torchvision datasets are PILImage images of range [0,1].
    We transform them to Tensors of normalized range [-1,1].
    """
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10_data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./CIFAR10_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    """
    Let us show some of the training images, for fun
    """
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


    """
    2. Define a Convolutional Neural Network
    
    Copy the neural network from the Neural Networks section before and modify it to take 3-channel images
    (instead of 1-channel images as it was defined)
    """
    net = Net()


    """
    3. Define a Loss function and optimizer
    
    Let's use a Classification Cross-Entropy loss and SGD with momentum
    """
    criterion = torch.nn.CrossEntropyLOss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    """
    4. Train the network
    
    This is when things start to get interesting. We simply have to loop over our data iterator,
    and feed the inputs to the network and optimize.
    """