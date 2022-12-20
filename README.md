# CS762-Knowledge-Distillation-Activation-Maps

# How to run this code?
* In order to train models we suggest using Google Cloud Pltaform (GCP) instances with GPU. GCP also greatly simplifies the environment setup since it comes with preinstalled libraries - when creating instance simply select Deep Learning VM Image. Once you have an instance, to train the model run following command:
```
cd niting_GCP
python main.py --model_dir [../experiments/resnet18_distill_Dec1]
```
Modify model_dir argument depending on which model you would like to train.
* In order to analyze models simply import notebook 762_heatmap.ipynb in Google Colab. The notebook contains all the necessary steps to produce saliency maps and heatmap. The notebook also already contains the commands to clone github repo and download the data from specific urls.

# Trained Models
https://drive.google.com/drive/folders/1yo87WHUJs0eJLX1rWJcYSAIJ4oO7F_WA?usp=share_link


