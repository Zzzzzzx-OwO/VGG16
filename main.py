import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("mps")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#def imshow(img):
 #   img = img/2+0.5
  #  npimg = img.numpy()
   # plt.imshow(np.transpose(npimg,(1,2,0)))
    #plt.show()
#images,labels = next(iter(trainloader))

#imshow(torchvision.utils.make_grid(images))
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

cfg = {
       'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
       def __init__(self, vgg_name):
           super(VGG, self).__init__()
           self.features = self._make_layers(cfg[vgg_name])
           self.classifier = nn.Linear(512, 10)
       def forward(self, x):
           out = self.features(x)
           out = out.view(out.size(0), -1)
           out = self.classifier(out)
           return out
       def _make_layers(self, cfg):
           layers = []
           in_channels = 3
           for x in cfg:
               if x == 'M':
                   layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
               else:
                   layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                              nn.BatchNorm2d(x),
                              nn.ReLU(inplace=True)]
                   in_channels = x
           layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
           return nn.Sequential(*layers)
vgg = VGG("VGG16")
vgg = vgg.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.parameters(),lr=0.001,momentum=0.9)

for epoch in range(10):
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = vgg(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if i%2000==1999:
            print('[%d,%5d] loss:%.3f' %(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0
print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
           images, labels = data
           images, labels = images.to(device), labels.to(device)
           outputs = vgg(images)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))




