'''
Functions for loading data, applying data transformation,
data augmentation and preprocessing of images
'''

import torch
from torchvision import datasets, transforms, models

from PIL import Image


def load_transform_data(data_dir):

    # TODO COMPLETED: Define your transforms for the training, validation, and testing sets
    # Training data augmentation (torchvision transforms are used to augment the training data
    # with random scaling, rotations, mirroring, and/or cropping)
    # Data normalization (The training, validation, and testing data is appropriately cropped and normalized)
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # TODO COMPLETED: Load the datasets with ImageFolder
    # Data loading (The data for each set (train, validation, test) is loaded with torchvision's ImageFolder)
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_valid_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_valid_transforms)

    # TODO COMPLETED: Using the image datasets and the trainforms, define the dataloaders
    # Data batching (The data for each set is loaded with torchvision's DataLoader)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    return train_data, test_data, valid_data, trainloader, testloader, validloader


# Image Processing (The process_image function successfully converts a PIL image
# into an object that can be used as input to a trained model)
def process_image(image_dir):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO COMPLETED: Process a PIL image for use in a PyTorch model
    loaded_image_using_pil = Image.open(f'{image_dir}')

    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    image_tensor = transform(loaded_image_using_pil)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor
