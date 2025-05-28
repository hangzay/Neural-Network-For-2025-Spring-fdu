import os

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

import pickle
from matplotlib import pyplot as plt


########## IMPLEMENT THE CODE BELOW, COMMENT OUT IRRELEVENT CODE IF NEEDED ##########
##### Model Definition #####
# TODO: Week 5, Task 1
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # TODO：在这里定义模型的结构。主要依赖于nn.Conv2d和nn.Linear
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO：在这里定义前向传播过程。这里输入的x形状是[Batch, 1, 28, 28]
        # x: (batch_size, 1, 28, 28)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # x: (batch_size, 6, 14, 14)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # x: (batch_size, 16, 5, 5)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        # x: (batch_size, 120)
        x = torch.relu(self.fc1(x))
        # x: (batch_size, 10)
        o = self.fc2(x)
        
        return o


##### Model Evaluation #####
# TODO: Week 5, Task 2
def evaluate(imgs, labels, model):
    # TODO：用model预测imgs
    # 获取最大值索引作为预测标签
    with torch.no_grad():
        pred_label = torch.argmax(model(imgs), dim=1)

    # TODO：计算预测正确的标签数量
    correct_cnt = torch.sum(pred_label == labels).item()
    
    print(f'match rate: {correct_cnt/labels.shape[0]}')
    return pred_label


##### Adversarial Attacks #####
# TODO: Week 5, Task 2
def fgsm(imgs, epsilon, model, criterion, labels):
    # 将模型切换至评估模式（关闭Dropout、BatchNorm等训练专用层），确保生成对抗样本时模型参数不被意外更新
    model.eval()

    adv_xs = imgs.float()
    adv_xs.requires_grad = True

    # TODO：模型前向传播，计算loss，然后反传
    # 输入对抗样本初始值（即原始图像）到模型中，获取预测结果
    outputs = model(adv_xs)
    # 计算损失函数，使用交叉熵损失函数
    loss = criterion(outputs, labels)
    # 反向传播计算梯度
    loss.backward()

    # TODO：得到输入的梯度、生成对抗样本
    grad = adv_xs.grad
    # torch.sign(grad)将梯度转换为±1的符号矩阵，确保扰动方向统一（最大化损失变化）
    # 通过参数epsilon控制扰动强度
    perturbation = epsilon * torch.sign(grad)
    adv_xs = adv_xs + perturbation

    # TODO：对扰动做截断，保证对抗样本的像素值在合理域内
    adv_xs = torch.clamp(adv_xs, 0, 1)

    model.train()

    return adv_xs.detach()


# TODO: Week 6, Task 1
def pgd(imgs, epsilon, iter, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()

    for i in range(iter):
        # Forward and compute loss, then backward

        # Retrieve grad and generate adversarial example, note to detach

        # Clip perturbation
        pass
    
    model.train()

    return adv_xs.detach()


# TODO: Week 6, Task 2
def fgsm_target(imgs, epsilon, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()
    adv_xs.requires_grad = True

    # Forward and compute loss, then backward

    # Retrieve grad and generate adversarial example, note to detach
    # Note to compute TARGETED loss and the sign of the perturbation

    # Clip perturbation

    model.train()

    return adv_xs.detach()

# TODO: Week 6, Task 2
def pgd_target(imgs, epsilon, iter, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()

    for i in range(iter):
        adv_xs.requires_grad = True
        
        # Forward and compute loss, then backward

        # Retrieve grad and generate adversarial example, note to detach
        # Note to compute TARGETED loss and the sign of the perturbation

        # Clip perturbation
    
    model.train()

    return adv_xs.detach()


########## NO NEED TO MODIFY CODE BELOW ##########
##### Data Loader #####
def load_mnist(batch_size):
    if not os.path.exists('data/'):
        os.mkdir('data/')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = torchvision.datasets.MNIST(root='data/', transform=transform, train=True, download=True)
    test_set = torchvision.datasets.MNIST(root='data/', transform=transform, train=False, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


##### Visualization #####
def visualize_benign(imgs, labels):
    fig = plt.figure(figsize=(8, 7))
    for idx, (img, label) in enumerate(zip(imgs, labels)):
        ax = fig.add_subplot(4, 5, idx + 1)
        ax.imshow(img[0], cmap='gray')
        ax.set_title(f'label: {label.item()}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def visualize_adv(imgs, true_labels, pred_labels):
    fig = plt.figure(figsize=(8, 8))
    for idx, (img, true_label, pred_label) in enumerate(zip(imgs, true_labels, pred_labels)):
        ax = fig.add_subplot(4, 5, idx + 1)
        ax.imshow(img[0], cmap='gray')
        ax.set_title(f'true label: {true_label.item()}\npred label: {pred_label.item()}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def visualize_target_adv(imgs, target_labels, pred_labels):
    fig = plt.figure(figsize=(8, 8))
    for idx, (img, true_label, pred_label) in enumerate(zip(imgs, target_labels, pred_labels)):
        ax = fig.add_subplot(4, 5, idx + 1)
        ax.imshow(img[0], cmap='gray')
        ax.set_title(f'target label: {true_label.item()}\npred label: {pred_label.item()}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
