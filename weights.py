'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2) # shuffle was True

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Load net from checkpoint
torch.nn.Module.dump_patches = True
checkpoint = torch.load('./checkpoint/ckpt18.t7')
net = checkpoint['net']
state_dict = net.state_dict()
state_keys_list = list(net.state_dict().keys())

# Get list of names, as named in the Keras model
def get_names():
    names = []
    names.append('conv1')
    names.append('bn1')

    for i in range(1, 5):
        for j in range(2):
            conv_name_base = 'conv' + str(i) + chr(ord('a')+j) + '_branch'
            bn_name_base = 'bn' + str(i) + chr(ord('a')+j) + '_branch'
            names.append(bn_name_base + '2a')
            names.append(conv_name_base + '2a')
            names.append(bn_name_base + '2b')
            names.append(conv_name_base + '2b')

            if i != 1 and j != 1:
                names.append(conv_name_base + '1')

    names.append('dense')
    return names

# Get list of parameters
def get_params():
    params = []
    cur_index = 0
    params.append(get_conv_weights(cur_index))
    cur_index += 1
    params.append(get_bn_params(cur_index))
    cur_index += 4

    for i in range(1, 5):
        for j in range(2):
            params.append(get_bn_params(cur_index))
            cur_index +=4
            params.append(get_conv_weights(cur_index))
            cur_index += 1
            params.append(get_bn_params(cur_index))
            cur_index +=4
            params.append(get_conv_weights(cur_index))
            cur_index += 1
            
            if i != 1 and j != 1:
                params.append(get_conv_weights(cur_index))
                cur_index += 1

    params.append(get_linear_params(cur_index))

    return params


def get_conv_weights(index):
    w = state_dict[state_keys_list[index]].cpu()
    return [np.transpose(w.numpy(), (2, 3, 1, 0)), np.zeros(w.numpy().shape[0])]

def get_bn_params(index):
    # state dict order for BN parameters are: weight, bias, running mean, & running variance
    w = state_dict[state_keys_list[index]].cpu().numpy()
    b = state_dict[state_keys_list[index + 1]].cpu().numpy()
    rm = state_dict[state_keys_list[index + 2]].cpu().numpy() 
    rv = state_dict[state_keys_list[index + 3]].cpu().numpy()  
    return [w, b, rm, rv]

def get_linear_params(index):
    w = state_dict[state_keys_list[index]].cpu()
    b = state_dict[state_keys_list[index + 1]].cpu()
    return [np.transpose(w.numpy(), (1, 0)), b.numpy()]

# Create dictionary where key = module name and value = module parameter 
def get_params_dictionary():
    names = get_names()
    params = get_params()
    params_dictionary = {name: param for name, param in zip(names, params)}
    return params_dictionary

# Construct pytorch model and load correct weights
model = ResNet18()
partial_model_dict = model.state_dict()

state_dict_subset = {k: v for k,v in state_dict.items() if k in partial_model_dict}
partial_model_dict.update(state_dict_subset)
model.load_state_dict(state_dict_subset)
print(model.state_dict().keys())
print(net.state_dict().keys())
model.cuda()

# model = net

count = 0
outputs = []
for i, data in enumerate(trainloader):
    x, y = data
    if use_cuda:
        x, y = x.cuda(), y.cuda() 
    inputs = Variable(x)
    output = model(inputs)
    output = output.cpu().data.numpy()
    outputs.append(output)
    count += 1
    if count == 1:
        break

def get_outputs():
    return np.vstack(outputs)

class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        self.start = nn.Sequential(
            *list(net.children())[:2]
        )
        self.layer1_0 = list(list(net.children())[2].children())[0] # nn.Sequential(
        # self.bn1 = self.layer1_0.bn1
        # self.conv1 = self.layer1_0.conv1
        # self.bn2 = self.layer1_0.bn2
        # self.conv2 = self.layer1_0.conv2

        # #     *list(list(net.children())[2].children())[:1]
        # # )
        # self.layer1_1 = list(list(net.children())[2].children())[1]

    def forward(self, x):
        x = F.relu(self.start(x))
        # out = F.relu(self.bn1(x))
        # shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # out = self.conv1(out)
        # out = self.conv2(F.relu(self.bn2(out)))
        # out += shortcut
        out = self.layer1_0(x)
        return out

model2 = my_model()
# print(model2)
count = 0
outputs2 = []
for i, data in enumerate(trainloader):
    x, y = data
    if use_cuda:
        x, y = x.cuda(), y.cuda() 
    inputs = Variable(x)
    output = model2(inputs)
    output = output.cpu().data.numpy()
    # print(output[0])
    outputs2.append(output)
    count += 1
    if count == 1:
        break

def get_outputs2():
    return np.vstack(outputs2)

# if use_cuda:
#     net.cuda()
#     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#     cudnn.benchmark = True

# criterion = nn.CrossEntropyLoss()
# # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

# # Training
# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda()
#         optimizer.zero_grad()
#         inputs, targets = Variable(inputs), Variable(targets)
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.data[0]
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         correct += predicted.eq(targets.data).cpu().sum()

#         progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(testloader):
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda()
#         inputs, targets = Variable(inputs, volatile=True), Variable(targets)
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)

#         test_loss += loss.data[0]
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         correct += predicted.eq(targets.data).cpu().sum()

#         progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
#     acc = 100.*correct/total
#     return acc


# for epoch in range(start_epoch, start_epoch+150):
#     train(epoch)
#     acc = test(epoch)

# # Save checkpoint.

# print('Saving..')
# state = {
#     'net': net.module if use_cuda else net,
#     'acc': acc,
#     'epoch': 149,
# }
# if not os.path.isdir('checkpoint'):
#     os.mkdir('checkpoint')
# torch.save(state, './checkpoint/ckpt50.t7')
# best_acc = acc
