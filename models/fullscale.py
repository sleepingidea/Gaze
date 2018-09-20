import torch
import torch.nn as nn


class InputBlock(nn.Module):

    def __init__(self, cfg):
        super(InputBlock, self).__init__()
        self.cfg = cfg
        self.bn1 = nn.BatchNorm2d(self.cfg[0])
        self.conv1s = nn.ModuleList([nn.Conv2d(self.cfg[1][1][i], self.cfg[1][3][i], kernel_size=3, padding=1) for i in range(self.cfg[1][0])])
        channels = sum(self.cfg[1][3])
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn1(x)
        c = []
        for i, conv1 in enumerate(self.conv1s):
            c.append(conv1(x[:, self.cfg[1][2][i]:self.cfg[1][2][i] + self.cfg[1][1][i], :, :]))
        x = torch.cat(c, 1)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


def test_InputBlock():
    device = torch.device("cuda")

    print("Testing InputBlock ...")
    input = torch.randn(2, 64, 192, 256).to(device)
    print(input.size())
    output = InputBlock([64, [3, [32, 32, 32], [0, 16, 32], [64, 64, 64]]]).to(device)(input)
    print(output.size())


class BasicBlock(nn.Module):

    def __init__(self, cfg):
        super(BasicBlock, self).__init__()
        self.cfg = cfg
        self.conv1s = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(self.cfg[1][1][i]), nn.Conv2d(self.cfg[1][1][i], self.cfg[1][3][i], kernel_size=5 if i == 1 else 3, padding=2 if i == 1 else 1)) for i in range(self.cfg[1][0])])
        channels = sum(self.cfg[1][3])
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c = []
        for i, conv1 in enumerate(self.conv1s):
            c.append(conv1(x[:, self.cfg[1][2][i]:self.cfg[1][2][i] + self.cfg[1][1][i], :, :].contiguous()))
        x = torch.cat(c, 1)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


def test_BasicBlock():
    device = torch.device("cuda")

    print("Testing BasicBlock ...")
    input = torch.randn(2, 192, 188, 252).to(device)
    print(input.size())
    output = BasicBlock([192, [2, [96, 96], [0, 96], [96, 96]]]).to(device)(input)
    print(output.size())


class BottleBlock(nn.Module):

    def __init__(self, cfg):
        super(BottleBlock, self).__init__()
        self.cfg = cfg
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv2d(self.cfg[0], self.cfg[1][0][0], kernel_size=1, padding=0),
                                                   nn.BatchNorm2d(self.cfg[1][0][0])),
                                     nn.Sequential(nn.Conv2d(self.cfg[0], self.cfg[1][1][0], kernel_size=1, padding=0),
                                                   nn.BatchNorm2d(self.cfg[1][1][0]),
                                                   nn.Conv2d(self.cfg[1][1][0], self.cfg[1][1][1], kernel_size=3, padding=1),
                                                   nn.BatchNorm2d(self.cfg[1][1][1])),
                                     nn.Sequential(nn.Conv2d(self.cfg[0], self.cfg[1][2][0], kernel_size=1, padding=0),
                                                   nn.BatchNorm2d(self.cfg[1][2][0]),
                                                   nn.Conv2d(self.cfg[1][2][0], self.cfg[1][2][1], kernel_size=5, padding=2),
                                                   nn.BatchNorm2d(self.cfg[1][2][1])),
                                     nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                                   nn.Conv2d(self.cfg[0], self.cfg[1][3][0], kernel_size=1, padding=0),
                                                   nn.BatchNorm2d(self.cfg[1][3][0])),
                                     ])

    def forward(self, x):
        c = []
        for i, conv in enumerate(self.convs):
            c.append(conv(x))
        x = torch.cat(c, 1)

        return x


def test_BottleBlock():
    device = torch.device("cuda")

    print("Testing BottleBlock ...")
    input = torch.randn(2, 192, 93, 125).to(device)
    print(input.size())
    output = BottleBlock([192, [[128], [128, 128], [128, 128], [128]]]).to(device)(input)
    print(output.size())


class ResBranch(nn.Module):

    def __init__(self, cfg):
        super(ResBranch, self).__init__()
        self.cfg = cfg
        self.resblocks = nn.ModuleList([self._make_resblock(self.cfg[1 + i]) for i in range(self.cfg[0])])
        self.bn1 = nn.BatchNorm2d(cfg[self.cfg[0]+1][0][0])
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.cfg[self.cfg[0]+1][0][0], self.cfg[self.cfg[0]+1][0][1], kernel_size=self.cfg[self.cfg[0]+1][0][2], stride=self.cfg[self.cfg[0]+1][0][3])
        self.bn2 = nn.BatchNorm2d(cfg[self.cfg[0]+1][1][0])
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.cfg[self.cfg[0]+1][1][0], self.cfg[self.cfg[0]+1][1][1], kernel_size=self.cfg[self.cfg[0]+1][1][2], stride=self.cfg[self.cfg[0]+1][1][3])
        self.bn3 = nn.BatchNorm2d(self.cfg[self.cfg[0]+1][1][1])
        self.relu3 = nn.ReLU(inplace=True)

    def _make_resblock(self, cfg):
        layers = []
        layers += [InputBlock(cfg[0])]
        layers += [self._make_block(cfg[1])]
        layers += [self._make_block(cfg[2])]
        layers += [self._make_block(cfg[3])]
        layers += [nn.ConvTranspose2d(cfg[4][0], cfg[4][1], kernel_size=cfg[4][2], stride=cfg[4][3], padding=cfg[4][4])]
        layers += [nn.ConvTranspose2d(cfg[5][0], cfg[5][1], kernel_size=cfg[5][2], stride=cfg[5][3], padding=cfg[5][4])]
        return nn.Sequential(*layers)

    def _make_block(self, cfg):
        return nn.Sequential(BasicBlock(cfg[0]), BottleBlock(cfg[1]))

    def forward(self, x):
        for i, resblock in enumerate(self.resblocks):
            residual = x
            x = resblock(x)
            x += residual

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x


class FullScale(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(FullScale, self).__init__()
        self.cfg = cfg
        self.opt = kwargs['opt']
        self.branch1 = ResBranch(cfg[0])
        self.branch2 = ResBranch(cfg[1])
        self.branch3 = ResBranch(cfg[2])
        self.dropout = nn.Dropout(cfg[3][0])
        self.linear1 = nn.Linear(cfg[3][1], cfg[3][2][0])
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(cfg[3][2][0], cfg[3][2][1])
        self.linear3 = nn.Linear(cfg[3][2][1], cfg[3][2][2])

    def _make_features(self):
        pass

    def forward(self, x):
        x = torch.cat((self.branch1(x), self.branch2(x), self.branch3(x)), 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x


cfgs = {
    '': [
        [3, [[1, [3, [1, 1, 1], [0, 0, 0], [8, 8, 8]]],
          [[24, [2, [12, 12], [0, 12], [16, 16]]], [32, [[16], [16, 16], [16, 16], [16]]]],
          [[64, [2, [16, 48], [0, 16], [64, 64]]], [128, [[64], [64, 64], [64, 64], [64]]]],
          [[256, [2, [64, 192], [0, 64], [128, 128]]], [256, [[128], [128, 128], [128, 128], [128]]]],
          [512, 128, 1, 1, (0, 0)],
          [128, 1, 1, 1, (0, 0)]],
         [[1, [3, [1, 1, 1], [0, 0, 0], [8, 8, 8]]],
          [[24, [2, [12, 12], [0, 12], [16, 16]]], [32, [[16], [16, 16], [16, 16], [16]]]],
          [[64, [2, [16, 48], [0, 16], [64, 64]]], [128, [[64], [64, 64], [64, 64], [64]]]],
          [[256, [2, [64, 192], [0, 64], [128, 128]]], [256, [[128], [128, 128], [128, 128], [128]]]],
          [512, 128, 1, 1, (0, 0)],
          [128, 1, 1, 1, (0, 0)]],
         [[1, [3, [1, 1, 1], [0, 0, 0], [8, 8, 8]]],
          [[24, [2, [12, 12], [0, 12], [16, 16]]], [32, [[16], [16, 16], [16, 16], [16]]]],
          [[64, [2, [16, 48], [0, 16], [64, 64]]], [128, [[64], [64, 64], [64, 64], [64]]]],
          [[256, [2, [64, 192], [0, 64], [128, 128]]], [256, [[128], [128, 128], [128, 128], [128]]]],
          [512, 128, 1, 1, (0, 0)],
          [128, 1, 1, 1, (0, 0)]],
         [[1, 8, 3, 2], [8, 4, 3, 2]]
         ],

        [2, [[1, [3, [1, 1, 1], [0, 0, 0], [8, 8, 8]]],
          [[24, [2, [12, 12], [0, 12], [16, 16]]], [32, [[16], [16, 16], [16, 16], [16]]]],
          [[64, [2, [16, 48], [0, 16], [64, 64]]], [128, [[64], [64, 64], [64, 64], [64]]]],
          [[256, [2, [64, 192], [0, 64], [128, 128]]], [256, [[128], [128, 128], [128, 128], [128]]]],
          [512, 128, 1, 1, (0, 0)],
          [128, 1, 1, 1, (0, 0)]],
         [[1, [3, [1, 1, 1], [0, 0, 0], [8, 8, 8]]],
          [[24, [2, [12, 12], [0, 12], [16, 16]]], [32, [[16], [16, 16], [16, 16], [16]]]],
          [[64, [2, [16, 48], [0, 16], [64, 64]]], [128, [[64], [64, 64], [64, 64], [64]]]],
          [[256, [2, [64, 192], [0, 64], [128, 128]]], [256, [[128], [128, 128], [128, 128], [128]]]],
          [512, 128, 1, 1, (0, 0)],
          [128, 1, 1, 1, (0, 0)]],
         [[1, 8, 3, 2], [8, 4, 3, 2]]
         ],

        [1, [[1, [3, [1, 1, 1], [0, 0, 0], [8, 8, 8]]],
          [[24, [2, [12, 12], [0, 12], [16, 16]]], [32, [[16], [16, 16], [16, 16], [16]]]],
          [[64, [2, [16, 48], [0, 16], [64, 64]]], [128, [[64], [64, 64], [64, 64], [64]]]],
          [[256, [2, [64, 192], [0, 64], [128, 128]]], [256, [[128], [128, 128], [128, 128], [128]]]],
          [512, 128, 1, 1, (0, 0)],
          [128, 1, 1, 1, (0, 0)]],
         [[1, 8, 3, 2], [8, 4, 3, 2]]
         ],

        [0.5, 1248, [64, 64, 3]]
         ]
}


def fullscale(opt, **kwargs):
    model = FullScale(cfgs[opt], opt=kwargs['cfg'])
    return model


def test_fullscale(cuda):
    import config

    device = torch.device("cuda" if cuda else "cpu")

    for m in list(cfgs.keys()):
        print("Testing FullScale_" + (m if m != '' else 'base') + " ...")
        cfg = eval("config.fullscale" + ('_' if m != '' else '') + m + "()")
        cfg.cuda = cuda
        input = torch.randn(1, 1, 35, 55)
        print(input.size())
        input = input.to(device)
        model = eval("fullscale('" + m + "', cfg=cfg).to(device)")
        output = model(input)
        print(output.size())
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(params)


if __name__ == "__main__":
    test_fullscale(True)
