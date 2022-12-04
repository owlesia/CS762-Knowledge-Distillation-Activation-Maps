import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from functorch import jacrev, vmap
from functorch.experimental import replace_all_batch_norm_modules_
import torchvision

num_classes = 37


def Resnet(model_name, pretrained=True):
    '''
    model_name : resnet50, resnet18 etc.
    '''
    # Load Model
    model = getattr(torchvision.models, model_name)(pretrained=pretrained)
    # model = torch.hub.load('pytorch/vision:v0.12.0',
    #                        model_name,
    #                        pretrained=pretrained)

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


class KD_loss():

    def __init__(self, inputs, outputs, labels, teacher_outputs, params,
                 teacher_model, student_model):
        self.params = params
        self.teacher = teacher_model
        self.student = student_model
        self.inputs = inputs
        self.outputs = outputs  # student outputs
        self.teacher_outputs = teacher_outputs
        self.labels = labels

    def __call__(self):
        if self.params.distill_loss_reg:
            return self.total_loss()
        else:
            return self.loss_kd()

    def loss_kd(self):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        alpha = self.params.alpha
        T = self.params.temperature
        KD_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(self.outputs/T, dim=1),
                                                      F.softmax(self.teacher_outputs/T, dim=1)) * (alpha * T * T) + \
            F.cross_entropy(self.outputs, self.labels) * (1. - alpha)

        return KD_loss

    def loss_regularised_kd(self):
        '''
        Compute loss with regularised term
        '''
        student_grad = self.get_gradients(self.student)
        teacher_grad = self.get_gradients(self.teacher).detach()

        reg_loss = nn.L1Loss()(student_grad, teacher_grad)
        #reg_loss = ((teacher_grad - student_grad)**2).mean()
        return reg_loss

    def total_loss(self):

        l = 1e+3
        w = self.params.reg_weight #weight of regularisation term
        kd_loss = self.loss_kd()
        reg_loss = l*self.loss_regularised_kd()
        total_loss = (1-w)*kd_loss + w*reg_loss
        return kd_loss, reg_loss, total_loss

    def get_gradients(self, model):
        '''
        Get gradients of the model wrt to the inputs
        '''
        replace_all_batch_norm_modules_(model)
        batch_grad = vmap(jacrev(predict, argnums=0), in_dims=(0, None, 0))
        dx = batch_grad(self.inputs, model, self.labels.unsqueeze(dim=1))
        return dx

def predict(input, model, label):
    '''
    predict the final model output for the true label
    '''
    input = input.unsqueeze(dim=0)
    out = model(input)
    out = out.squeeze(dim=0)

    # pick the output for the true label
    out = out[label]
    return out