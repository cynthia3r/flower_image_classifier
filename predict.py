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

parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='set device mode cuda or cpu (default=cuda)')

results = parser.parse_args()

image_dir = results.image_dir
model_input = results.model_input
top_k = results.top_k

if results.device == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA GPU mode is enabled now')
elif results.device == 'cpu':
    device = torch.device('cpu')
    print('CPU mode is enabled now')
else:
    print('CUDA GPU is not supported on this system so switching to CPU mode')
    device = torch.device('cpu')

category_names = results.category_names
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

model, optimizer = load_checkpoint(model_input)

for param in model.parameters():
    param.requires_grad = False


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
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model.forward(image_tensor)

    ps = torch.exp(output)
    # Top K classes: the function predicts the top K classes along with associated probabilities
    probs_topk = np.array(ps.topk(topk)[0])[0]
    class_idx = np.array(ps.topk(topk)[1])[0]

    # Invert the model.class_to_idx dict to obtain mapping
    idx_to_class = dict((value, key) for key, value in model.class_to_idx.items())
    class_topk = []
    for idx in class_idx:
        class_topk.append(idx_to_class[idx])

    return image_tensor, probs_topk, class_topk


# Predicting classes: The predict function successfully reads in an image and a checkpoint
# then returns the most likely image class index and it's associated probability
# Run prediction function
image_tensor, probs_topk, class_topk = predict(image_dir, model, top_k, device)

predicted_flower_names = [cat_to_name[idx] for idx in class_topk]
print("{} most likely predicted flower names = {}".format(top_k, predicted_flower_names))
print("{} most likely predicted probability = {}".format(top_k, probs_topk))
print("The model predicted flower name as {} with {:.3f}% probability".format(predicted_flower_names[0], 100 * probs_topk[0]))
