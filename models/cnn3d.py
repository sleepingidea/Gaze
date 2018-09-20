import torch.nn as nn
import math


class CNN3D(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(CNN3D, self).__init__()
        self.cfg = cfg
        self.opt = kwargs['opt']
        self.features = self._make_features(self.cfg[0])
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * (m.in_channels + m.out_channels)
                m.weight.data.normal_(0, math.sqrt(4. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.in_features + m.out_features
                m.weight.data.normal_(0, math.sqrt(4. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_features(self, cfg):
        layers = []
        for v in cfg:
            if isinstance(v, int):
                layers += [nn.BatchNorm3d(v)]
            elif len(v) == 5:
                conv3d = nn.Conv3d(v[0], v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                layers += [conv3d, nn.BatchNorm3d(v[1]), nn.ReLU(inplace=True)]
            elif len(v) == 2:
                layers += [nn.MaxPool3d(kernel_size=v[0], stride=v[1])]
        return nn.Sequential(*layers)


cfgs = {
    # '3': [[1, [1, 64, (17, 33, 65), (2, 1, 1), 0], [64, 64, (9, 17, 33), (2, 1, 1), 0], [64, 1, (9, 17, 33), 2, 0]]],
}


def cnn3d(opt, **kwargs):
    model = CNN3D(cfgs[opt], opt=kwargs['cfg'])
    return model


def test_cnn3d(cuda):
    import config
    import torch

    device = torch.device("cuda" if cuda else "cpu")

    for m in list(cfgs.keys()):
        print("Testing CNN3D_" + (m if m != '' else 'base') + " ...")
        cfg = eval("config.cnn3d" + ('_' if m != '' else '') + m + "()")
        cfg.cuda = cuda
        input = torch.randn(1, 1, cfg.kernel.kT, cfg.kernel.kW, cfg.kernel.kH)
        print(input.size())
        input = input.to(device)
        model = eval("cnn3d('" + m + "', cfg=cfg).to(device)")
        output = model(input)
        print(output.size())
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(params)


if __name__ == "__main__":
    test_cnn3d(True)
