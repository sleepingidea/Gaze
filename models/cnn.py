import torch.nn as nn
import math


class CNN(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(CNN, self).__init__()
        self.opt = kwargs['opt']
        self.features = self._make_features(cfg[1], cfg[0])
        self.classifier = self._make_classifier(cfg[3], cfg[2], cfg[4])
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels + m.out_channels)
                m.weight.data.normal_(0, math.sqrt(4. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.in_features + m.out_features
                m.weight.data.normal_(0, math.sqrt(4. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_features(self, cfg, in_channels):
        layers = []
        for v in cfg:
            if v == 'B':
                layers += [nn.BatchNorm2d(in_channels)]
            elif isinstance(v, str) and v[0] == 'D':
                v = int(v[1:])
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=2)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _make_classifier(self, cfg, in_features, num_classes):
        layers = []
        for v in cfg:
            layers += [nn.Linear(in_features, v), nn.ReLU(inplace=True)]
            in_features = v
        layers += [nn.Linear(in_features, num_classes)]
        return nn.Sequential(*layers)


cfgs = {
    '7': [1, ['B', 8, 8, 16, 'D16', 'D32', 'D32'], 1120, [], 3],
    '12': [1, ['B', 8, 8, 8, 16, 16, 16, 'D32', 'D32', 'D32'], 1120, [128, 128], 3],
}


def cnn(opt, **kwargs):
    model = CNN(cfgs[opt], opt=kwargs['cfg'])
    return model


def test_cnn(cuda):
    import config
    import torch

    device = torch.device("cuda" if cuda else "cpu")

    for m in list(cfgs.keys()):
        print("Testing CNN_" + (m if m != '' else 'base') + " ...")
        cfg = eval("config.cnn" + ('_' if m != '' else '') + m + "()")
        cfg.cuda = cuda
        input = torch.randn(2, 1, 35, 55)
        print(input.size())
        input = input.to(device)
        model = eval("cnn('" + m + "', cfg=cfg).to(device)")
        output = model(input)
        print(output.size())
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(params)


if __name__ == "__main__":
    test_cnn(True)
