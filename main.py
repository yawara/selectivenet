import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from tensorboardX import SummaryWriter

from selectivenet.selectivenet import SelectiveNet
from selectivenet.resnet import resnet10


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, 128)

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

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def train(args, model, device, train_loader, writer, optimizer, epoch):
    model.train()

    train_empirical_coverage = 0
    train_r = 0
    train_l2_loss_coverage = 0
    train_loss_fg = 0
    train_loss_h = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        f, g, h = model(data)

        empirical_coverage = g.mean()
        r = (F.nll_loss(F.log_softmax(f), target, reduction='none') * g).mean() / empirical_coverage
        l2_loss_coverage = (args.coverage - empirical_coverage) ** 2
        loss_fg = r + args.l * l2_loss_coverage
        loss_h = F.nll_loss(F.log_softmax(h), target)
        loss = args.alpha * loss_fg + (1-args.alpha) * loss_h

        train_empirical_coverage += empirical_coverage.item()
        train_r += r.item()
        train_l2_loss_coverage += l2_loss_coverage.item()
        train_loss_fg += loss_fg.item()
        train_loss_h += loss_h.item()
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_empirical_coverage /= len(train_loader)
    train_r /= len(train_loader)
    train_l2_loss_coverage /= len(train_loader)
    train_loss_fg /= len(train_loader)
    train_loss_h /= len(train_loader)
    train_loss /= len(train_loader)

    writer.add_scalar('train/empirical_coverage', train_empirical_coverage, epoch)
    writer.add_scalar('train/r', train_r, epoch)
    writer.add_scalar('train/l2_loss_coverage', train_l2_loss_coverage, epoch)
    writer.add_scalar('train/loss_fg', train_loss_fg, epoch)
    writer.add_scalar('train/loss_h', train_loss_h, epoch)
    writer.add_scalar('train/loss', train_loss, epoch)


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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--coverage', type=float, default=0.85)
    parser.add_argument('--l', type=float, default=32.)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--logdir')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(args.logdir)

    train_dataset = CIFAR10(args.root, train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(args.root, train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    feature = Net()
    model = SelectiveNet(feature, feature_dim=128, num_classes=10)
    model.to(device)

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, writer, optimizer, epoch)
        test(args, model, device, test_loader, writer, epoch)


if __name__ == "__main__":
    main()
