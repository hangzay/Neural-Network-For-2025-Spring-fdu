import os

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

import pickle
from matplotlib import pyplot as plt


########## IMPLEMENT THE CODE BELOW, COMMENT OUT IRRELEVENT CODE IF NEEDED ##########
##### Model Definition #####
# TODO: Week 5, Task 1 (请迁移Week 5已实现代码)
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
    pred_label = torch.argmax(model(imgs), dim=1)

    # TODO：计算预测正确的标签数量
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
        with torch.no_grad():
            output = model(img) 
            pred_label = output.argmax(dim=1)  
            correct_cnt += (pred_label == label).sum().item() 
            sample_cnt += label.size(0)

        t.set_postfix(test_acc=correct_cnt/sample_cnt)


##### Adversarial Attacks #####
# TODO: Week 5, Task 2
def fgsm(imgs, epsilon, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()
    adv_xs.requires_grad = True

    # TODO：模型前向传播，计算loss，然后反传
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


# TODO: Week 6, Bonus (请迁移Week 6已实现代码，或注释)
def nes(imgs, epsilon, model, labels, sigma, n):
    """
    labels: ground truth labels
    sigma: search variance
    n: number of samples used for estimation for each img
    """
    model.eval()

    adv_xs = imgs.reshape(-1, 28 * 28).float()

    grad = torch.zeros_like(adv_xs)
    # TODO: Estimate gradient for each sample adv_x in adv_xs
    batch_size = adv_xs.size(0)
    device = adv_xs.device
    
    # 生成高斯噪声 [batch, n, 784]
    noise = torch.randn(batch_size, n, 28 * 28, device=device)
    
    # 创建正负扰动样本 [2*batch, n, 784]
    perturbed = torch.cat([
        adv_xs.unsqueeze(1) + sigma * noise,
        adv_xs.unsqueeze(1) - sigma * noise
    ], dim=0).view(-1, 1, 28, 28)
    
    # 获取模型预测概率
    with torch.no_grad():
        probs = model(perturbed).softmax(dim=1)
    
    # 提取正确标签的概率 [2*batch*n]
    labels_expanded = labels.repeat_interleave(n).repeat(2)
    correct_probs = probs[torch.arange(2*batch_size*n), labels_expanded]
    
    # 分割正负扰动结果 [batch, n]
    pos_probs = correct_probs[:batch_size*n].view(batch_size, n)
    neg_probs = correct_probs[batch_size*n:].view(batch_size, n)
    
    # 计算NES梯度估计 [batch, 784]
    grad = ((pos_probs - neg_probs) / (2*sigma)).unsqueeze(-1) * noise
    grad = grad.mean(dim=1)

    adv_xs = adv_xs.detach() - epsilon * grad.sign()
    adv_xs = torch.clamp(adv_xs, min=0., max=1.)

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
