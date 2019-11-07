from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import os
import cv2



class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, int(output_channels/4), 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(int(output_channels/4))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(output_channels/4), int(output_channels/4), 3, stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(int(output_channels/4))
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(int(output_channels/4), output_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False)
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out



class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, size1, size2, size3):
        super(AttentionModule, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)
        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax3_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.softmax4_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )
        self.last_blocks = ResidualBlock(in_channels, out_channels)
    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 =  self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        #
        out_interp3 = self.interpolation3(out_softmax3)
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4)
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5)
        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = (1 + out_softmax6) * out_trunk
        out_last = self.last_blocks(out)
        return out_last





class ResidualAttentionModel(nn.Module):
    def __init__(self):
        super(ResidualAttentionModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule(256, 256, (56,56), (28,28), (14,14))
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule(512, 512, (28,28), (14,14), (7,7))
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule(1024, 1024, (14,14), (7,7), (4,4))
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,170)
    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


import tensorflow as tf
import pathlib
data_dir = '/home/celso//ResidualAttentionNetwork-pytorch/Residual-Attention-Network/downloads'
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
# Image Preprocessing 
transform = transforms.Compose([
    transforms.Scale(224),
    transforms.ToTensor()])

transform2 = transforms.Compose([
    transforms.Resize([224,224],),
    transforms.ToTensor()])

train_dataset2 = datasets.CIFAR10(root='./data/',
                               train=True, 
                               transform=transform,
                               download=True)

test_dataset2 = datasets.CIFAR10(root='./data/',
                              train=False, 
                              transform=transform2)

train_data = datasets.ImageFolder('/home/celso//ResidualAttentionNetwork-pytorch/Residual-Attention-Network/downloads', transform=transform2)
train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

#gerenciamento de informaçoes dinamico e




# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=20, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=20, 
                                          shuffle=False)

classes = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])


model = ResidualAttentionModel().cuda()
print(model)

lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training 
for epoch in range(100):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        # print(images.data)
        labels = Variable(labels.cuda())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #print("step i = ",i)
        if (i+1) % 100 == 0:
            print ("Epoch [",epoch+1,"/","80], Iter [",i+1,"/",500,"] Loss: ",loss.data)
    # Decaying Learning Rate
    if (epoch+1) % 20 == 0:
        lr /= 3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

# Save the Model
torch.save(model.state_dict(), 'model.pkl')

# Test
correct = 0
total = 0
#
class_correct = list(0. for i in range(170))
class_total = list(0. for i in range(170))

for images, labels in test_loader:
    images = Variable(images.cuda())
    labels = Variable(labels.cuda())
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()
    #
    c = (predicted == labels.data).squeeze()
    for i in range(4):
        label = labels.data[i]
        class_correct[label] += c[i]
        class_total[label] += 1

print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# for data in testloader:
#     images, labels = data
#     outputs = model(Variable(images.cuda()))
#     _, predicted = torch.max(outputs.data, 1)
#     c = (predicted == labels).squeeze()
#     for i in range(4):
#         label = labels[i]
#         class_correct[label] += c[i]
#         class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))