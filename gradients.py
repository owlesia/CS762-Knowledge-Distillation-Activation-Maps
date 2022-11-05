from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def input_gradient(x, model):
    pass


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


def compare_saliency_maps(sm1, sm2):
    return torch.abs(sm1 - sm2).sum()
