# PASCAL VOC Object Classification

![giphy](https://user-images.githubusercontent.com/21276946/55680190-e706a300-5948-11e9-9e32-de2797be3b87.gif) 

## Problem Statement
The goal of this project is to recognize objects from a number of visual object classes in realistic scenes. There are 20 object classes:
1. Person
2. Bird, cat, cow, dog, horse, sheep
3. Aeroplane, bicycle, boat, bus, car, motorbike, train
4. Bottle, chair, dining table, potted plant, sofa, tv/ monitor

Specifically for the classification task, the goal is, for each of the classes predict the presence/ absence of at least one object of that class in a test image.

## Data
We will use Pascal VOC 2012 dataset for this project and the latest version of pytorch has Pascal VOC dataset class built-in. For the purpose of this project, we will only use training set and validation set of Pascal VOC. The ground truth annotation for the dataset contains the following information,
* Class: the object class. I.e. car or person
* Bounding box: an axis-aligned rectangle specifying the extent of the object visible in the image
* View: ‘frontal’ , ‘rear’, ‘left’ or right
* Difficult: an object marked as difficult indicates that the object is considered difficult to recognize without substantial use of context.
* Truncated: an object marked as ‘truncated’ indicates that the bounding box specified for the object does not correspond to the full extent of the object.
* Occluded: an object marked as ‘occluded’ indicates that a significant portion of the  subject image is within the bounding box occluded by another object.

For our task, we regarded all ‘difficult’ marked objects as negative examples.

## Loss function

The task is multi-label classification for 20 object classes, which is analogous to creating 20 object detectors, 1 for every class. Hence we have used binary cross entropy (with logits loss) as our loss function.

In pytorch, the loss function is torch.nn.BCEWithLogitsLoss( ). Do note that this function provides numerical stability over the sequence of sigmoid followed by binary cross entropy. The loss function is clearly documented at ***https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#BCEWithLogitsLoss***

## Metrics
We used average precision as the metric to measure performance. It is simply the average of the maximum precisions at different recall values. 

Accuracy is a very poor metric to be used in this problem. I.e.: In a picture containing a person and dog, the model can output the result as train and get away with an accuracy of 85%!

## Model
We used ResNet50 as our deep learning architecture due to relatively low memory utilization since number of parameters are less. (We used our local machines to train, preserving AWS credits for final project).  

We used transfer learning method since the object classes are very similar to ImageNet classes.

## Hyper-parameters  
| Hyperparameter          | Values                                                 |
|-------------------------|--------------------------------------------------------|
| Image Size              | 300x300                                                |
| Batch size              | 32                                                     |
| Initial lr              | [1e-5 for resnet backbone, 5e-3 for the fclayer]       |
| Optimizer               | SGD                                                    |
| Learning rate scheduler | CosineAnnealing learning rate scheduler with Tmax = 12 |
| Momentum                | 0.9                                                    |
| Epochs                  | 15                                                     |

## Challenges
Hyperparameter search was an interesting challenge we faced. Initially our model faced overfitting problems and we were able to fix the problem by,
1. Using smaller learning rate for the pre-trained resnet backbone so that we do not disrupt imagenet weights drastically.
2. Using larger learning rate for the randomly initialized fully connected layer.
3. Using learning rate scheduler rather than using static learning rates.
4. Choosing good set of image augmentations to add small amount of noise during training to make the model robust.

## Results
### Training History
![loss-1](https://user-images.githubusercontent.com/21276946/55679999-d3f2d380-5946-11e9-92a4-45b3f0356ab9.png)
![accuracy-1](https://user-images.githubusercontent.com/21276946/55680003-d7865a80-5946-11e9-80a4-14898c4c88e7.png)

### Mean Tail Accuracy vs Classification Thresholds
The graph below shows the variation of mean tail accuracies against classification thresholds for the entire validation dataset for 20 equally spaced threshold values from 0 to 1.0. 

![clf_vs_threshold-1](https://user-images.githubusercontent.com/21276946/55679982-a4dc6200-5946-11e9-950d-c5ef48e1a4f9.png)

## Flask Application for Inference 
We built a flask application to allow users to predict on new images. The HTML page is rendered on the server and displayed in a browser. A user can select and upload a file using the website. Which will enable a predict button, that will display the predictions of the network when pressed. The repository containing the Flask Web application can be found at ***https://github.com/sk-aravind/Pytorch-Flask-Webapp***

## How to reproduce the code?
### Pytorch source code
1. Install dependencies: pip install -r requirements.txt
2. Directory structure
    * /docs: contain project and devkit documentation
    * /models: contains model weights, log-files and plots
    * /src: contains source code
    * /data: data directory to download and extract pascal VOC dataset (You should create this directory manually)
    * Run the main function in main.py with required arguments. The codebase is clearly documented with clear details on how to execute the functions. It also includes an example. You need to interface only with this function to reproduce the code.

### Flask App
1. Install dependencies : pip install -r requirements.txt
2. Deploy the app : python deploy.py
3. Open a web browser and go to http://localhost:8000
    * The CSV that renders the table is stored in static/csv
    * The Pascal VOC Image Data is stored in static/data
