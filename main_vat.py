import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip
from tensorboardX import SummaryWriter


from selectivenet.vat import VAT


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        # (32, 16, 16)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        # (64, 8, 8)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        # (128, 4, 4)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        # (256, 2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, inplace=True)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, inplace=True)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.leaky_relu(x, inplace=True)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.leaky_relu(x, inplace=True)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def train(args, model, device, train_loader, writer, optimizer, epoch, vat):
    model.train()
    train_loss = 0
    train_likelihood_loss = 0
    train_vat_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        likelihood_loss = F.nll_loss(output, target)
        vat_loss = vat.forward(model, data)
        loss = likelihood_loss + args.vat_lambda * vat_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_likelihood_loss += likelihood_loss.item()
        train_vat_loss += vat_loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

    train_loss /= len(train_loader)
    train_likelihood_loss /= len(train_loader)
    train_vat_loss /= len(train_loader)

    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/vat_loss', train_vat_loss, epoch)
    writer.add_scalar('train/likelihood_loss', train_likelihood_loss, epoch)


def test(args, model, device, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, _, output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output), target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    test_acc = correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    writer.add_scalar('test/loss', test_loss, epoch)
    writer.add_scalar('test/acc', test_acc, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--vat_lambda', type=float, default=1.0)
    parser.add_argument('--vat_epsilon', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--logdir')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(args.logdir)

    train_transform = Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])
    test_transform = ToTensor()
    train_dataset = CIFAR10(args.root, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10(args.root, train=False, transform=test_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    model = Net()
    model.to(device)

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
    vat = VAT(args.vat_epsilon)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, writer, optimizer, epoch, vat)
        test(args, model, device, test_loader, writer, epoch)
        lr_scheduler.step()


if __name__ == "__main__":
    main()
