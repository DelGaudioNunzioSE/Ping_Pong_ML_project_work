"""

    Machine Learning Project Work: Tennis Table Tournament
    Group 2:
        Ciaravola Giosuè - g.ciaravola3#studenti.unisa.it
        Conato Christian - c.conato@studenti.unisa.it
        Del Gaudio Nunzio - n.delgaudio5@studenti.unisa.it
        Garofalo Mariachiara - m.garofalo38@studenti.unisa.it

    ---------------------------------------------------------------

    arm_supervised_train.py

    File containing the supervised learning for controlling
    the arm movements.

"""

import sys
import os

# Get the current and parent directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)  # Add the parent directory to the system path

from utilities.action_space import ActionSpaceArm
import torch
import csv
from utilities.early_stopping import EarlyStopping
from nets.arm_net import ArmModel
from utilities.dataset_loader import ArmDataset
import torch.nn as nn
from torch.optim import Adam

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
NUM_EPOCHS = 1000
HIDDEN_SIZE = (100, 50)
NUM_INPUTS = 2

# Initialize EarlyStopping
early_stopping = EarlyStopping()

# Initialize the action space and model
action_space_arm = ActionSpaceArm()
arm_model = ArmModel(HIDDEN_SIZE, NUM_INPUTS, action_space_arm).to(device)

# Initialize the optimizer and loss function
optimizer = Adam(arm_model.parameters(), lr=0.001)
lossFunction = nn.MSELoss()

# Load the dataset
dataset = ArmDataset('dataset_file.csv')

# Open a CSV file to log the training and validation loss
with open("arm_learning_report.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'epoch', 'training_loss', 'val_loss'
    ])

# Training loop
final_epoch = 0
for epoch in range(NUM_EPOCHS):
    arm_model.train()  # Set the model to training mode
    training_loss = 0.0

    # Training phase
    for input_train_batch, target_train_batch in dataset.train_loader:
        input_train_batch, target_train_batch = input_train_batch.to(device), target_train_batch.to(device)
        optimizer.zero_grad()  # Clear the gradients
        joints = arm_model(input_train_batch)  # Forward pass
        loss = lossFunction(joints, target_train_batch)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        training_loss += loss.item() * input_train_batch.size(0)  # Accumulate the training loss

    arm_model.eval()
    val_loss = 0.0

    # Validation phase
    with torch.no_grad():
        for input_val_batch, target_val_batch in dataset.val_loader:
            input_val_batch, target_val_batch = input_val_batch.to(device), target_val_batch.to(device)
            joints = arm_model(input_val_batch)  # Forward pass
            loss = lossFunction(joints, target_val_batch)  # Compute the loss
            val_loss += loss.item() * input_val_batch.size(0)  # Accumulate the validation loss

    # Compute the average training and validation loss
    training_loss /= len(dataset.train_loader.dataset)
    val_loss /= len(dataset.val_loader.dataset)

    # Print the current epoch, training loss, and validation loss
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Log the losses to the CSV file
    with open("arm_learning_report.csv", 'a') as file:
        file.write(str(epoch) + ',' + str(training_loss) + ',' + str(val_loss) + '\n')

    # Check for early stopping
    early_stopping(val_loss, arm_model)
    if early_stopping.early_stop:
        final_epoch = epoch
        print('Early stopping triggered')
        break

# Save the final model checkpoint
arm_model.save_checkpoint(final_epoch, 'arm_model_dataset_bello')
print('Training finished.')
