import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as D
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchattacks
import os
from PIL import Image
import numpy as np
import pickle
import string
import argparser
from tqdm import tqdm

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='training epochs')
parser.add_argument('--val_ratio', default=0.2, type=float, help='ratio of validation data')
parser.add_argument('--resume', action='store_true', help='if resume from checkpoint')
parser.add_argument('--model_path', type=str, help='model saving path')
parser.add_argument('--data_path', type=str, help='data dir')
parser.add_argument('--adv_path', type=str, help='adversarial outputs dir')
args = parser.parse_args()

# classes did not exclude 0 1 o l
classes = []
for i in range(10):    classes.append(str(i))
for i in string.ascii_lowercase:    classes.append(i)

# data processing: 5/6 for training 1/6 for adversarial
data = []
with open(args.data_path + 'data1.pkl', 'rb') as F:    data1 = pickle.load(F)
with open(args.data_path + 'data2.pkl', 'rb') as F:    data2 = pickle.load(F)
data += data1 + data2
with open(args.data_path + 'label.pkl', 'rb') as F:    label = pickle.load(F)

print(len(data), len(label))
adv_data = data[200000:]
adv_label = label[200000:]
data = data[:200000]
label = label[:200000]

np.random.seed(19990814)
val_idx = np.arange(len(data))
np.random.shuffle(val_idx)
val_idx = val_idx[:int(len(data)*args.val_ratio)]

train_data, test_data, train_label, test_label = [], [], [], []
for i in range(len(data)):
    if i in val_idx:
        test_data.append(data[i])
        test_label.append(label[i])
    else:
        train_data.append(data[i])
        train_label.append(label[i])

# resize image into (32,32), data shape: (batch_size, 3, 32, 32)
class trainset(D.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.label[i]
    
    def collate_fn(inputs):
        image = [nn.functional.interpolate(entry[0].unsqueeze(0), size = (32,32)) for entry in inputs]
        label = [classes.index(entry[1]) for entry in inputs]
        return torch.cat(image), torch.tensor(label)
    
    def collate_fn_adv(inputs):
        image = [entry[0] for entry in inputs]
        label = [entry[1] for entry in inputs]
        return torch.stack(image), torch.stack(label)

best_acc = 0
start_epoch = 0

trainloader = D.DataLoader(dataset = trainset(train_data, train_label),
                           batch_size=128,
                           shuffle=True,
                           collate_fn = trainset.collate_fn,
                           num_workers=0)
testloader = D.DataLoader(dataset = trainset(test_data, test_label),
                           batch_size=256,
                           shuffle=False,
                           collate_fn = trainset.collate_fn,
                           num_workers=0)

net = models.vgg19_bn(pretrained=True)
net.classifier[6] = nn.Linear(4096,36)

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('./drive/My Drive/colab_spml_final/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(best_acc)

net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print("loss: %.3f, Acc: %.3f" % (train_loss/(batch_idx+1), 100.*correct/total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print("loss: %.3f, Acc: %.3f" % (test_loss/(batch_idx+1), 100.*correct/total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, args.model_path)
        best_acc = acc

for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    test(epoch)