# CS762-Knowledge-Distillation-Activation-Maps

1. Knowledge Distillation
2. Evaluation / Interpretability
3. Brainstorm Novel Way of Knowledge Distillation Considering Interpretability


# KD Resources

- https://josehoras.github.io/knowledge-distillation/ <br />
Straightforward tutorial with code available in Jupyter Notebook https://github.com/josehoras/Knowledge-Distillation/blob/master/knowledge_distillation.ipynb. Uses MNIST dataset and some linear NNs (not ResNet). Copied code to our repo at path KD_examples/Example1_Josehoras. To start go to notebook knowledge_distillation.ipynb. Made changes in notebook to run on cpu, and also the notebook cell where we can train the teacher model ourselves (not required to do that) had multiple errors, fixed them.

- https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/knowledge_distillation.ipynb <br />
Also straightforward colab notebook. Uses MNIST dataset and some linear NNs (not ResNet). 

- https://github.com/yoshitomo-matsubara/torchdistill <br />
Some complicated framework for model distillation, not sure we can use it. It has good examples of different resnet models though: https://github.com/yoshitomo-matsubara/torchdistill/blob/main/torchdistill/models/classification/resnet.py. 

- https://github.com/Adlik/model_optimizer <br />
Another framework that seems complicated to use.

- https://github.com/haitongli/knowledge-distillation-pytorch <br />
Average difficulty to understand the repo, more on the easy side. It has code implemented for training a ResNet-18 model with knowledge distilled from a pre-trained ResNext-29 teacher. We probably could use it.

- https://github.com/Incremental-Learning/Knowledge-Distillation-Keras-1/blob/master/Knowledge_Distillation_Notebook.ipynb

- https://wandb.ai/authors/knowledge-distillation/reports/Distilling-Knowledge-in-Neural-Networks--VmlldzoyMjkxODk <br />
Colab Notebook.

- https://keras.io/examples/keras_recipes/better_knowledge_distillation/ <br />
Official Keras KD documentation leads to this doc: https://github.com/sayakpaul/FunMatch-Distillation. 


