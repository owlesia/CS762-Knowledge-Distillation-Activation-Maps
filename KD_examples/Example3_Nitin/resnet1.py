import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

num_classes = 37


def Resnet(model_name, pretrained=True):
    '''
    model_name : resnet50, resnet18 etc.
    '''
    # Load Model
    model = torch.hub.load('pytorch/vision:v0.12.0',
                           model_name,
                           pretrained=pretrained)

    # Change the last linear layer as per our num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if pretrained:
        torch.nn.init.xavier_uniform_(model.fc.weight)
    else:
        logging.info("Initializing Weights")
        model.apply(initialize_weights)

    return model


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(outputs/T, dim=1),
                                                  F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
        F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels)/float(labels.size)


# maintain all metrics required in this dictionary-
# these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}


def initialize_weights(m):
    
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
