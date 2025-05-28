import os

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

import pickle
from matplotlib import pyplot as plt


########## IMPLEMENT THE CODE BELOW, COMMENT OUT IRRELEVENT CODE IF NEEDED ##########
##### Model Definition #####
# Week 5, Task 1 (请迁移Week 5已实现代码)
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
# Week 5, Task 2 (请迁移Week 5已实现代码)
def evaluate(imgs, labels, model):
    # TODO：用model预测imgs，并得到预测标签pred_label
    pred_label = torch.argmax(model(imgs), dim=1)

    # TODO：计算预测标签pred_label与真实标签labels的匹配数目
    correct_cnt = torch.sum(pred_label == labels).item()
    
    print(f'match rate: {correct_cnt/labels.shape[0]}')
    return pred_label

# TODO: Week 7, Task 1
def evaluate_dataloader(dataloader, model):
    model.eval()
    
    correct_cnt, sample_cnt = 0, 0

    t = tqdm(dataloader)
    for img, label in t:
        # TODO: Predict label for img, update correct_cnt, sample_cnt
        output = model(img)
        pred_label = torch.argmax(output, dim=1)

        correct_cnt += torch.sum(pred_label == label).item()
        sample_cnt += img.shape[0]

        t.set_postfix(test_acc=correct_cnt/sample_cnt)

##### Adversarial Attacks #####
# Week 5, Task 2 (请迁移Week 5已实现代码)
def fgsm(imgs, epsilon, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()
    adv_xs.requires_grad = True

    # TODO：模型前向传播，计算loss，然后loss反传
    outputs = model(adv_xs)
    loss = criterion(outputs, labels)
    loss.backward()

    # TODO：得到输入的梯度、生成对抗样本
    grad = adv_xs.grad.data
    perturbation = epsilon * torch.sign(grad)
    adv_xs = adv_xs + perturbation

    # TODO：对扰动做截断，保证对抗样本的像素值在合理域内
    adv_xs = torch.clamp(adv_xs, 0, 1)

    model.train()

    return adv_xs.detach()


# TODO: Week 6, Task 1
def pgd(imgs, epsilon, alpha, iter, model, criterion, labels):
    model.eval()

    # 从原图开始迭代，初始扰动可为0
    adv_xs = imgs.float()

    for i in range(iter):

        adv_xs.requires_grad = True 

        # Forward and compute loss, then backward
        outputs = model(adv_xs)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        # Retrieve grad and generate adversarial example, note to detach
        grad = adv_xs.grad
        adv_xs = adv_xs + alpha * grad.sign()
        perturbation = torch.clamp(adv_xs - imgs, -epsilon, epsilon)

        # Clip perturbation
        adv_xs = torch.clamp(imgs + perturbation, 0, 1)

        # 获取一个脱离计算图的tensor，获得的tensor不再会被反向计算梯度
        adv_xs = adv_xs.detach()

    model.train()

    return adv_xs.detach()


# TODO: Week 6, Task 2
def fgsm_target(imgs, epsilon, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()
    adv_xs.requires_grad = True

    # Forward and compute loss, then backward
    outputs = model(adv_xs)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()

    # Retrieve grad and generate adversarial example, note to detach
    # Note to compute TARGETED loss and the sign of the perturbation
    grad = adv_xs.grad
    adv_xs = adv_xs - epsilon * grad.sign()

    # Clip perturbation
    adv_xs = torch.clamp(adv_xs, 0, 1)

    model.train()

    return adv_xs.detach()

# TODO: Week 6, Task 2
def pgd_target(imgs, epsilon, alpha, iter, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()

    for i in range(iter):

        adv_xs.requires_grad = True 

        # Forward and compute loss, then backward
        outputs = model(adv_xs)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        # Retrieve grad and generate adversarial example, note to detach
        grad = adv_xs.grad
        adv_xs = adv_xs - alpha * grad.sign()
        perturbation = torch.clamp(adv_xs - imgs, -epsilon, epsilon)

        # Clip perturbation
        adv_xs = torch.clamp(imgs + perturbation, 0, 1)

        adv_xs = adv_xs.detach()

    model.train()

    return adv_xs.detach()

# TODO: Week 6, Bonus
def nes(imgs, epsilon, model, labels, sigma, n):
    """
    NES黑盒攻击实现
    imgs: [batch, 1, 28, 28]，原始图片
    epsilon: 扰动强度
    model: 被攻击模型
    labels: [batch]，目标标签
    sigma: NES采样噪声标准差
    n: 采样次数
    """
    model.eval()
    device = imgs.device
    batch_size = imgs.size(0)
    adv_xs = imgs.clone().detach()

    # 初始化累计梯度
    grad = torch.zeros_like(adv_xs)

    for i in range(n):
        # 采样高斯噪声
        noise = torch.randn_like(adv_xs)
        x_plus = adv_xs + sigma * noise
        x_minus = adv_xs - sigma * noise

        # 拼接批量
        x_cat = torch.cat([x_plus, x_minus], dim=0)  # [2*batch, 1, 28, 28]
        with torch.no_grad():
            logits = model(x_cat)
            probs = torch.softmax(logits, dim=1)
        # 取目标类别概率
        labels_cat = torch.cat([labels, labels], dim=0)
        idx = torch.arange(2 * batch_size, device=device)
        prob_y = probs[idx, labels_cat]

        prob_plus = prob_y[:batch_size]
        prob_minus = prob_y[batch_size:]

        # NES梯度估计
        grad += ((prob_plus - prob_minus) / (2 * sigma)).view(-1, 1, 1, 1) * noise

    grad = grad / n

    # 生成对抗样本
    adv_xs = adv_xs - epsilon * grad.sign()
    adv_xs = torch.clamp(adv_xs, 0.0, 1.0)
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
