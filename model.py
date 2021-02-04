import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from arch import ResNet20
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os
from visualize import plot_losses, plot_accuracies, plot_weights, plot_clustered_distributions
from prune import prune_weights, ReadAllWeights



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
prune_coeff = 0.50


print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

print('==> Building model..')
net = ResNet20()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train(epoch, net):
    print('\nEpoch: %d' % epoch)
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    train_epoch_losses = []
    train_epoch_accs = []
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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
        
        train_epoch_loss = train_loss/(batch_idx+1)
        train_epoch_acc = 100.*correct/total
        
        train_epoch_losses.append(train_epoch_loss)
        train_epoch_accs.append(train_epoch_acc)
        
    print('Loss: {:.3f} | Acc: {:.3f}'.format(train_epoch_loss, train_epoch_acc))
    return np.array(train_epoch_losses), np.array(train_epoch_accs)


def test(epoch, net, checkpoint_path='./checkpoint/ckpt.pth'):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    test_epoch_losses = []
    test_epoch_accs = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            test_epoch_loss = test_loss/(batch_idx+1)
            test_epoch_acc = 100.*correct/total
            
            test_epoch_losses.append(test_epoch_loss)
            test_epoch_accs.append(test_epoch_acc)
            
    print('Loss: {:.3f} | Acc: {:.3f}'.format(test_epoch_loss, test_epoch_acc))
            

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
        torch.save(state, checkpoint_path)
        best_acc = acc
    
    return np.array(test_epoch_losses), np.array(test_epoch_accs)


def build_model(net, epochs_num=2):

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs_num):

        train_epoch_losses, train_epoch_accs = train(epoch, net)
        test_epoch_losses, test_epoch_accs = test(epoch, net)

        train_losses.append(np.mean(train_epoch_losses))   # добавляем усреднённое по батчам
        train_accs.append(np.mean(train_epoch_accs))
        test_losses.append(np.mean(test_epoch_losses))
        test_accs.append(np.mean(test_epoch_accs))

        scheduler.step()

    plot_losses(train_losses, test_losses)
    plot_accuracies(train_accs, test_accs)


init_model = build_model(net)

# Load weights from checkpoint.
print('==> Resuming from checkpoint..')
checkpoint = torch.load('/home/user/Desktop/task2/checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])


# prune weights
# The pruned weight location is saved in the addressbook and maskbook.
# These will be used during training to keep the weights zero.
addressbook = []
maskbook = []
for k, v in net.state_dict().items():
    print(k)
    if "conv2" in k:
        addressbook.append(k)
        print("pruning layer:", k)
        weights = v
        weights, masks = prune_weights(weights, prune_coeff)
        maskbook.append(masks)
        checkpoint['net'][k] = weights

checkpoint['address'] = addressbook
checkpoint['mask'] = maskbook
net.load_state_dict(checkpoint['net'])
pruned_model = build_model(net)
module_name, layer_name, weights = ReadAllWeights(net)
plot_weights(module_name, layer_name, weights)
plot_clustered_distributions(weights)