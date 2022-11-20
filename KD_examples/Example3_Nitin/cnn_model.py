import torch
from torch import nn


def freeze_all(model_params):
    for param in model_params:
        param.requires_grad = False


def unfreeze_all(model_params):
    for param in model_params:
        param.requires_grad = True


bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def head_blocks(in_dim, p, out_dim, activation=None):
    "Basic Linear block"
    layers = [
        nn.BatchNorm1d(in_dim),
        nn.Dropout(p),
        nn.Linear(in_dim, out_dim)
    ]

    if activation is not None:
        layers.append(activation)

    return layers


def create_head(nf, nc, bn_final=False):
    "Model head that takes in 'nf' features and outputs 'nc' classes"
    pool = AdaptiveConcatPool2d()
    layers = [pool, nn.Flatten()]
    layers += head_blocks(nf, 0.25, 512, nn.ReLU(inplace=True))
    layers += head_blocks(512, 0.5, nc)

    if bn_final:
        layers.append(nn.BatchNorm1d(nc, momentum=0.01))

    return nn.Sequential(*layers)


def requires_grad(layer):
    "Determines whether 'layer' requires gradients"
    ps = list(layer.parameters())
    if not ps:
        return None
    return ps[0].requires_grad


def cnn_model(model, num_classes, bn_final=False, init=nn.init.kaiming_normal_):
    "Creates a model using a pretrained 'model' and appends a new head to it with 'nc' outputs"

    nf = model.fc.in_features

    # remove dense and freeze everything
    body = nn.Sequential(*list(model.children())[:-2])
    
    head = create_head(nf, num_classes, bn_final)

    model = nn.Sequential(body, head)

    # freeze the base of the model
    freeze_all(model[0].parameters())

    # initialize the weights of the head
    for child in model[1].children():
        if isinstance(child, nn.Module) and (not isinstance(child, bn_types)) and requires_grad(child):
            init(child.weight)

    return model


def get_cnn_model(model_name='resnet_18', num_classes=37):

    resnet = torch.hub.load('pytorch/vision:v0.10.0',
                            model_name, pretrained=True)
    #model = cnn_model(resnet, num_classes, bn_final=True)
    nf = resnet.fc.in_features
    resnet.fc = nn.Linear(nf, num_classes)
    return resnet
