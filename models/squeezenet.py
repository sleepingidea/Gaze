import torch
import torch.nn as nn


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze_bn(self.squeeze(x)))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1_bn(self.expand1x1(x))),
            self.expand3x3_activation(self.expand3x3_bn(self.expand3x3(x)))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(SqueezeNet, self).__init__()

        self.opt = kwargs['opt']
        self.version = cfg[0]
        self.num_classes = cfg[1]

        if self.version == 1.0:
            self.features = nn.Sequential(
                nn.BatchNorm2d(1),
                # nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(1, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.BatchNorm2d(1),
                # nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(1, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            # nn.ReLU(inplace=True),
            nn.AvgPool2d((8, 13), stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


cfgs = {
    'v10': [1.0, 3],
    'v11': [1.1, 3]
}


def squeezenet(opt, **kwargs):
    model = SqueezeNet(cfgs[opt], opt=kwargs['cfg'])
    return model


def test_squeezenet(cuda):
    import config

    device = torch.device("cuda" if cuda else "cpu")

    for m in list(cfgs.keys()):
        print("Testing SqueezeNet_" + (m if m != '' else 'base') + " ...")
        cfg = eval("config.squeezenet" + ('_' if m != '' else '') + m + "()")
        cfg.cuda = cuda
        input = torch.randn(2, 1, 35, 55)
        print(input.size())
        input = input.to(device)
        model = eval("squeezenet('" + m + "', cfg=cfg).to(device)")
        output = model(input)
        print(output.size())
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(params)


if __name__ == "__main__":
    test_squeezenet(True)
