import torch
from torchvision import models

import time
'''
Functions and classes related to training model
'''


def save_checkpoint(model, optimizer, train_data, arch, save_dir):
    # TODO COMPLETED: Save the checkpoint
    model.to('cpu')

    print("Our model: \n\n", model, '\n')
    print("The state dict keys: \n\n", model.state_dict().keys())

    # Saving the model (The trained model is saved as a checkpoint along with associated
    # hyperparameters and the class_to_idx dictionary)
    checkpoint = {'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'state_dict': model.state_dict(),
                  'arch': arch,
                  'optimizer': optimizer,
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, save_dir)


# TODO COMPLETED: Write a function that loads a checkpoint and rebuilds the model
# Loading checkpoints (The function successfully loads a checkpoint and rebuilds the model)
def load_checkpoint(model_input):
    checkpoint = torch.load(model_input)

    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']

    model.load_state_dict(checkpoint['state_dict'])

    # load the optimizer from saved checkpoint
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


# Define Validation function for the model
def validate_model(model, validloader, criterion, device):
    valid_loss = 0
    accuracy = 0

    model.to(device)

    # Looping through it, get a batch on each loop
    # validation pass here
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        # Forward pass, get our log-probabilities
        outputs = model.forward(images)
        # Calculate the loss with the calculated log-probabilities and the labels
        valid_loss += criterion(outputs, labels)

        # Calculate accuracy
        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
    return valid_loss, accuracy


# Training a network: successfully trains a new network on a dataset of images
# Model hyperparameters: The training model run based on user set hyperparameters such as training epochs
# Training with GPU: The training model allows users to choose training the model based on device (gpu, cpu)
def train_model(model, trainloader, validloader, criterion, num_epochs, optimizer, device):

    # TODO COMPLETED: Build and train your network
    # Training the network (The parameters of the feedforward classifier are appropriately trained,
    # while the parameters of the feature network are left static)

    train_losses, valid_losses = [], []

    model.to(device)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0

        for images, labels in trainloader:
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)

            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            # Forward pass, get our log-probabilities
            outputs = model.forward(images)
            # Calculate the loss with the calculated log-probabilities and the labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            # Validate model
            valid_loss = 0
            accuracy = 0

            # set model to evaluation mode for predictions
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validate_model(model, validloader, criterion, device)

            train_losses.append(running_loss / len(trainloader))
            valid_losses.append(valid_loss / len(validloader))

            # Training validation log: The training loss, validation loss, and
            # validation accuracy are printed out as a network trains
            print("Epoch: {}/{}.. ".format(epoch + 1, num_epochs),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Validation Loss: {:.3f}.. ".format(valid_losses[-1]),
                  "Validation Accuracy: {:.3f}%".format((100 * accuracy) / len(validloader)))

            # set model back to train mode
            model.train()

        # Calculate and print Epoch duration
        epoch_time_elapsed = time.time() - epoch_start_time
        print("Epoch {} Run Time: {:.0f}m {:.0f}s".format(epoch + 1, epoch_time_elapsed // 60, epoch_time_elapsed % 60))

    return model, optimizer


# TODO COMPLETED: Do validation on the test input data
def test_model(model, testloader, device):
    accuracy = 0

    model.to(device)

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        # set model to evaluation mode for predictions
        model.eval()

        # Looping through it, get a batch on each loop
        # test validation pass here
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)

            # Calculate accuracy
            # Get the class probabilities
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

            # set model back to train mode
            model.train()

        # Testing Accuracy (The network's accuracy is measured on the test data)
        print("Test Accuracy: {:.3f}%".format(100 * accuracy / len(testloader)))
