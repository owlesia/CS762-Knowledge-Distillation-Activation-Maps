from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import matplotlib
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def get_gradcam_saliency_maps(x, y, model):
  with GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=torch.cuda.is_available()) as cam:
      # We have to specify the target we want to generate
      # the Class Activation Maps for.
      # If targets is None, the highest scoring category
      # will be used for every image in the batch.
      # Here we use ClassifierOutputTarget, but you can define your own custom targets
      # That are, for example, combinations of categories, or specific outputs in a non standard model.

      targets = [ClassifierOutputTarget(y)]

      # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
      grayscale_cam = cam(input_tensor=x, targets=targets)

      # TODO: Do we want to use eigen_smooth=True in the above function to remove some of the noise??

  return grayscale_cam


# def compare_saliency_maps(sm1, sm2):
#     return torch.abs(sm1 - sm2).sum()

def compare_saliency_maps(sm1, sm2):
    return torch.abs(sm1 - sm2).sum()


def get_model_similarity_scores(model_list, dataloader, device, num_classes=1000, use_all_labels=False, loss_function=torch.nn.CrossEntropyLoss()):
  model_heatmap = np.array([[0 for i in range(len(models_list))] for j in range(len(models_list))], dtype=float)
  model_predictions_dict = {i:[] for i in range(len(models_list))}

  for model in model_list:
    model.eval()

  for batch in data_loader: # Get batch
    images, labels = batch # Unpack the batch into images and labels


    if use_all_labels:
      for label in range(num_classes):
        print(label)
        labels = torch.tensor([label]*len(labels))
        images, labels = images.to(device), labels.to(device)
        images.requires_grad_(requires_grad=True).retain_grad()
        labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).type(torch.float)

        for model_index in range(len(models_list)):

          if images.grad != None:
            images.grad.data.zero_()
          prediction = models_list[model_index](images)
          prediction = torch.nn.Softmax(dim=1)(prediction)
          loss = loss_function(labels, prediction)
          loss.backward()
          model_grad = images.grad.detach().cpu()

          model_predictions_dict[model_index] = model_grad

        for model_index1 in range(len(models_list)):
          for model_index2 in range(len(models_list)):
            sm1 = model_predictions_dict[model_index1]
            sm2 = model_predictions_dict[model_index2]
            model_heatmap[model_index1, model_index2] += compare_saliency_maps(sm1, sm2)/len(data_loader.dataset)

    else:
      images, labels = images.to(device), labels.to(device)
      images.requires_grad_(requires_grad=True).retain_grad()
      labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).type(torch.float)

      for model_index in range(len(models_list)):

        if images.grad != None:
          images.grad.data.zero_()
        prediction = models_list[model_index](images)
        prediction = torch.nn.Softmax(dim=1)(prediction)
        loss = loss_function(labels, prediction)
        loss.backward()
        model_grad = images.grad.detach().cpu()

        model_predictions_dict[model_index].append(model_grad)

      for model_index1 in range(len(models_list)):
        for model_index2 in range(len(models_list)):
          sm1 = torch.cat(model_predictions_dict[model_index1])
          sm2 = torch.cat(model_predictions_dict[model_index2])
          model_heatmap[model_index1, model_index2] += compare_saliency_maps(sm1, sm2)/len(data_loader.dataset)

    plt.imshow(model_heatmap)
    plt.title("model comparison heatmap")
    plt.colorbar()
    plt.show()

  return model_heatmap
