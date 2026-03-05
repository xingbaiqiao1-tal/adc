import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# --- 1. 核心 ADC 算子 ---
class ADC_Module(nn.Module):
    def __init__(self, channels, kernel_size=3, alpha=0.1):
        super(ADC_Module, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        # alpha 是可学习的，让网络自己决定需要多少差分信息
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), alpha))
        
    def forward(self, x):
        b, c, h, w = x.shape
        # 使用 unfold 提取滑窗，并动态获取尺寸防止 view 报错
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        l = x_unfold.size(-1)
        
        # 维度变换: [B, C*K2, L] -> [B, C, K2, L]
        x_unfold = x_unfold.view(b, c, self.kernel_size**2, l)
        
        # 提取中心像素进行差分
        center_idx = (self.kernel_size**2) // 2
        x_center = x_unfold[:, :, center_idx:center_idx+1, :]
        
        # 计算 sum|x_center - xi|
        # 原理类似于非线性梯度，突出边缘
        diff_sum = torch.sum(torch.abs(x_center - x_unfold), dim=2)
        diff_feat = diff_sum.view(b, c, h, w)
        
        return x + self.alpha * diff_feat

# --- 2. 带 ADC 的残差块 ---
class ADC_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ADC_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 加入 ADC 模块
        self.adc = ADC_Module(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.adc(out) # 差分增强
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# --- 3. ADC-ResNet18 主体结构 ---
class ADC_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ADC_ResNet, self).__init__()
        self.in_planes = 64

        # CIFAR-10 特化 Stem: 3x3 卷积代替 7x7，不使用 MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ADC_ResNet18(num_classes=10):
    return ADC_ResNet(ADC_BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 纯传统卷积，不包含 ADC
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CIFAR_ResNet_Baseline(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CIFAR_ResNet_Baseline, self).__init__()
        self.in_planes = 64

        # 关键改进：针对 32x32 输入，使用 3x3 卷积且 stride=1
        # 这确保了进入 Layer1 时分辨率依然是 32x32，没有信息丢失
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 移除了原始 ResNet 中的 MaxPool 层

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def get_cifar_resnet18():
    return CIFAR_ResNet_Baseline(BasicBlock, [2, 2, 2, 2])


def train3():
    # 1. 硬件配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BATCH_SIZE = 50
    # 2. 数据准备 (CIFAR-10)
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    train_dataset = torchvision.datasets.ImageFolder(root='./cdata/cifar100/trainval', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./cdata', train=False,
    #                                     download=False, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root='./cdata/cifar100/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2)

    # 3. 初始化模型
    # model = get_formula_a_resnet18(num_classes=10).to(device)
    # model = resnet18(num_classes=10).to(device)
    # model = resnet18_custom(num_classes=10).to(device)
    model = ADC_ResNet18(num_classes=100).to(device)
    # model = get_cifar_resnet18().to(device)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 建议先用 Adam 快速收敛，后期可以用 SGD + Momentum 冲击更高准确率
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 5. 训练循环
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # 6. 每个 Epoch 测试一次
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f'Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(trainloader):.4f} | Acc: {100.*correct/total:.2f}%')
        scheduler.step()
        


if __name__ == "__main__":
    train3()
