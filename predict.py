import argparse
import numpy as np
import json

import torch
from torchvision import datasets, transforms, models

from utility import process_image
from model import load_checkpoint

parser = argparse.ArgumentParser(description='Predict the top K most likely flower classes based on image path and saved checkpoint')

# Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
parser.add_argument('--image_dir', type=str, default='./flowers/test/74/image_01191.jpg',
                        help='set path of the flower image (default=./flowers/test/74/image_01191.jpg)')

parser.add_argument('--model_input', type=str, default='checkpoint.pth',
                        help='set path of the model checkpoint (default=checkpoint.pth)')

parser.add_argument('--top_k', type=int, default=5,
                        help='set number of top K most likely classes (default=5)')

parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='set file for mapping of flower categories to category names (default=cat_to_name.json)')

parser.add_argument('--device', type=str, default='cuda',
                        help='set device mode cuda or cpu (default=cuda)')

results = parser.parse_args()

image_dir = results.image_dir
model_input = results.model_input
top_k = results.top_k
category_names = results.category_names
device = results.device


model, optimizer = load_checkpoint(model_input)

for param in model.parameters():
    param.requires_grad = False

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)


# Class Prediction (The predict function successfully takes the path to an image and a checkpoint,
# then returns the top K most probably classes for that image)
# Predicting with GPU: The function allows users to use the GPU or CPU to calculate the predictions
def predict(image_dir, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO COMPLETED: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()

    image_tensor = process_image(image_dir)
    image_tensor = image_tensor.to('cuda')

    with torch.no_grad():
        output = model.forward(image_tensor)

    ps = torch.exp(output)
    # Top K classes: the function predicts the top K classes along with associated probabilities
    probs = np.array(ps.topk(topk)[0])[0]
    class_idx = np.array(ps.topk(topk)[1])[0]

    # Return preconverted numpy array from a tensor torch along with the image tensor
    return image_tensor, probs, class_idx


# Predicting classes: The predict function successfully reads in an image and a checkpoint
# then returns the most likely image class and it's associated probability
# Run prediction function
image_tensor, probs, class_idx = predict(image_dir, model, top_k, device)
print(probs)
print(class_idx)

# Retrieve the flowername from the cat_to_name dictionary
# Displaying class names: the script maps the class values to category names based on loaded json mapping in cat_to_name
idx_to_flower_name = {category: cat_to_name[str(category)] for category in class_idx}
predicted_flower_names = list(idx_to_flower_name.values())

print("Predicted Flower Category Name = {}".format(predicted_flower_names))
print("Predicted probability = {}".format(probs))
print("The model predicted flower name as {} with {}% probability".format(predicted_flower_names[0], 100 * probs[0]))
