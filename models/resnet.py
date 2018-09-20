import torch.nn as nn
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 8
        super(ResNet, self).__init__()

        self.opt = kwargs['opt']
        self.block = eval(cfg[0])
        self.layers = cfg[1]
        self.num_classes = cfg[2]

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, 8, self.layers[0])
        self.layer2 = self._make_layer(self.block, 16, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 32, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 64, self.layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((5, 7), stride=1)
        self.fc = nn.Linear(64 * self.block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


cfgs = {
    '18': ['BasicBlock', [2, 2, 2, 2], 3],
    # '34': ['BasicBlock', [3, 4, 6, 3], 64],
    # '50': ['Bottleneck', [3, 4, 6, 3], 64],
    # '101': ['Bottleneck', [3, 4, 23, 3], 64],
    # '152': ['Bottleneck', [3, 8, 36, 3], 64],
}


def resnet(opt, **kwargs):
    model = ResNet(cfgs[opt], opt=kwargs['cfg'])
    return model


def test_resnet(cuda):
    import config
    import torch

    device = torch.device("cuda" if cuda else "cpu")

    for m in list(cfgs.keys()):
        print("Testing ResNet_" + (m if m != '' else 'base') + " ...")
        cfg = eval("config.resnet" + ('_' if m != '' else '') + m + "()")
        cfg.cuda = cuda
        input = torch.randn(2, 1, 35, 55)
        print(input.size())
        input = input.to(device)
        model = eval("resnet('" + m + "', cfg=cfg).to(device)")
        output = model(input)
        print(output.size())
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(params)


if __name__ == "__main__":
    test_resnet(True)
