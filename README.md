# AI Programming with Python Final Project (Create Your Own Flower Image Classifier)

The project involves two parts, developing code for an image classifier using PyTorch and then conversion of the code into a command line Python application. VGG16 neural network model has been used for this project.

## Part 1 - Development of an Image Classifier with Deep Learning 

The first part of the project involves implementation of an image classifier through a Jupyter notebook using PyTorch.

### Specifications
For detailed project specifications please refer the Jupyter notebook ```Image Classifier Project.ipynb```

### Steps involved
1) Package Imports: All the necessary packages and modules are imported in the first cell of the notebook
2) Training data augmentation: torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping
3) Data normalization: The training, validation, and testing data is appropriately cropped and normalized
4) Data batching: The data for each set is loaded with torchvision's DataLoader
5) Data loading: The data for each set (train, validation, test) is loaded with torchvision's ImageFolder
6) Pretrained Network: A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen
7) Feedforward Classifier: A new feedforward network is defined for use as a classifier using the features as input
8) Training the network: The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static
9) Testing Accuracy: The network's accuracy is measured on the test data
10) Validation Loss and Accuracy: During training, the validation loss and accuracy are displayed
11) Loading checkpoints: There is a function that successfully loads a checkpoint and rebuilds the model
12) Saving the model: The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary
13) Image Processing: The process_image function successfully converts a PIL image into an object that can be used as input to a trained model
14) Class Prediction: The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image
15) Sanity Checking with matplotlib: A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names

### Usage
Run the Jupyter notebook in GPU enabled mode to build the model and use it for prediction

## Part 2 - Building the command line application

Now that the deep neural network model is built and trained on the flower data set, it's time to convert it into an application so that others can use it for prediction. The built application is a pair of Python scripts that run from the command line. For testing, the model checkpoint will be used that has been generated and saved in the first part of the project.

### Specifications
The project submission includes two files train.py and predict.py. The first file, train.py, trains a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. A separate file has been created for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images.

### Steps involved
1) Training a network: train.py successfully trains a new network on a dataset of images and saves the model to a checkpoint
2) Training validation log: The training loss, validation loss, and validation accuracy are printed out as a network trains
3) Model architecture: The training script allows users to choose from at least two different architectures available from torchvision.models
4) Model hyperparameters: The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
5) Training with GPU: The training script allows users to choose training the model on a GPU
6) Predicting classes: The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
7) Top K classes: The predict.py script allows users to print out the top K classes along with associated probabilities
8) Displaying class names: The predict.py script allows users to load a JSON file that maps the class values to other category names
9) Predicting with GPU: The predict.py script allows users to use the GPU to calculate the predictions

### Usage
1. Train
Train a new network on a data set with ```train.py```
    - Basic usage: ```python train.py data_directory```
    - Prints out training loss, validation loss, and validation accuracy as the network trains
    - Options:
        - Set directory to save checkpoints: ```python train.py data_dir --save_dir save_directory```
        - Choose architecture: ```python train.py data_dir --arch "vgg13"```
        - Set hyperparameters: ```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20```
        - Use GPU for training: ```python train.py data_dir –gpu```
2. Predict
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
    - Basic usage: ```python predict.py /path/to/image checkpoint```
    - Options:
        - Return top K most likely classes: ```python predict.py input checkpoint --top_k 3```
        - Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_to_name.json```
        - Use GPU for inference: ```python predict.py input checkpoint --gpu```
        
	Note: argparse module in the python standard library has been used to get the command line input into the scripts.

