from .cnn import *
from .capsule import *
from .resnet import *
from .squeezenet import *
from .fullscale import *
from .cnn3d import *


def create_model(model, **kwargs):
    m = model.split('_')
    return eval("" + m[0] + "('" + ("" if len(m) == 1 else m[1]) + "', cfg=kwargs['cfg'])")
