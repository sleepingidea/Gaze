import os


class base(object):

    def __init__(self):

        self._space = 0

    def __eq__(self, other):

        if isinstance(other, self.__class__):
            flag = True
            for name, value in vars(self).items():
                flag &= (value == eval('other.' + name))
            return flag
        else:
            return False

    def __repr__(self):

        s = []
        sp = []
        for i in range(self._space):
            sp.append(' ')

        if self._space == 0:
            s.extend(sp)
            s.append(self.__class__.__name__)
            s.append(': ')
        s.append('{\n')
        sp2 = sp.copy()
        sp2.append('  ')

        v = list(vars(self).items())
        v.sort()

        for name, value in v:
            if name in ['_space']:
                continue
            s.extend(sp2)
            s.append(name)
            s.append(': ')
            if value.__class__.__base__ == base:
                value._space = self._space + 2
            s.append(str(value))
            s.append('\n')
        s.extend(sp)
        s.append('}')
        if self._space == 0:
            s.append('\n')

        return ''.join(s)


__urls__ = [
    'https://en.myluo.cn/packages/gaze.zip',
]


class paths(base):

    def __init__(self, data_folder='datasets', h5_file='gaze.h5', json_file='gaze.json', save_folder='save',
                 check_file='_checkpoint.pth', ms_file='_ms.pth', logging_file='_logging.log'):
        super(paths, self).__init__()
        self.data_folder = data_folder
        self.h5_file = h5_file
        self.json_file = json_file
        self.save_folder = save_folder
        self.check_file = check_file
        self.ms_file = ms_file
        self.logging_file = logging_file


class Dataset(base):

    def __init__(self, batch_size, test_batch_size, epochs, lr, lr_decay, momentum, noise=0., data_ratio=0.5,
                 is3d=False, nonorm=False, nocuda=False, seed=1, ci=False, paths=paths(),
                 root_folder=os.path.abspath('.'), data_reset=False):
        super(Dataset, self).__init__()
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.noise = noise
        self.data_ratio = data_ratio
        self.is3d = is3d
        self.nonorm = nonorm
        self.nocuda = nocuda
        self.seed = seed
        self.ci = ci
        self.paths = paths
        self.root_folder = root_folder
        self.data_reset = data_reset


def resnet_18():
    dataset = Dataset(32, 32, 10, 0.01, 0, 0)
    return dataset


def resnet_34():
    dataset = Dataset(96, 96, 30, 0.01, 0, 0)
    return dataset


def resnet_50():
    dataset = Dataset(32, 32, 30, 0.01, 0, 0)
    return dataset


def resnet_101():
    dataset = Dataset(16, 16, 30, 0.01, 0, 0)
    return dataset


def resnet_152():
    dataset = Dataset(16, 16, 30, 0.01, 0, 0)
    return dataset


def cnn_7():
    dataset = Dataset(32, 25000, 10, 0.01, 0, 0)
    return dataset


def cnn_12():
    dataset = Dataset(32, 25000, 10, 0.01, 0, 0)
    return dataset


def capsule():
    dataset = Dataset(32, 32, 10, 0.01, 0, 0)
    return dataset


def capsule_w():
    dataset = Dataset(32, 32, 10, 0.01, 0, 0)
    return dataset


def capsule_7():
    dataset = Dataset(256, 256, 10, 0.01, 0, 0)
    return dataset


def capsule_12():
    dataset = Dataset(128, 128, 10, 0.01, 0, 0)
    return dataset


def capsule_3():
    dataset = Dataset(32, 32, 10, 0.01, 0, 0)
    return dataset


def capsule_3w():
    dataset = Dataset(192, 192, 6, 0.01, 0, 0)
    return dataset


def capsule_4():
    dataset = Dataset(128, 128, 50, 0.01, 0, 0)
    return dataset


def capsule_5():
    dataset = Dataset(512, 512, 10, 0.01, 0, 0)
    return dataset


def squeezenet_v10():
    dataset = Dataset(32, 1024, 10, 0.01, 0, 0)
    return dataset


def squeezenet_v11():
    dataset = Dataset(32, 1024, 10, 0.01, 0, 0)
    return dataset


def fullscale():
    dataset = Dataset(32, 32, 10, 0.01, 0, 0)
    return dataset


def cnn3d_3():
    dataset = Dataset(1, 1, 50, 0.01, 0, 0, is3d=True)
    return dataset


default = cnn_7()


def cfg(model=None):
    if model is not None:
        return eval("" + model + "()")
    else:
        return default


def args2dataset(args):
    dataset = Dataset(batch_size=args.batchsize, test_batch_size=args.testbatchsize, epochs=args.epochs, lr=args.lr,
                      lr_decay=args.lrdecay, momentum=args.momentum, noise=args.noise, data_ratio=args.ratio,
                      is3d=default.is3d, nonorm=default.nonorm, nocuda=args.no_cuda, seed=args.seed, ci=args.ci,
                      paths=default.paths, root_folder=default.root_folder, data_reset=args.reset)
    return dataset


if __name__ == "__main__":
    print(cfg())