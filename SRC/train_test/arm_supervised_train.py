import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

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

NUM_EPOCHS = 1000
early_stopping = EarlyStopping()
HIDDEN_SIZE = (100, 50)
NUM_INPUTS = 2
action_space_arm = ActionSpaceArm()
arm_model = ArmModel(HIDDEN_SIZE, NUM_INPUTS, action_space_arm).to(device)
optimizer = Adam(arm_model.parameters(), lr=0.001)
dataset = ArmDataset('dataset_file_2.csv')
lossFunction = nn.MSELoss()

with open("arm_learning_report_100_200_50_001_dataset_bello.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'epoch', 'training_loss', 'val_loss'
    ])

final_epoch = 0
for epoch in range(NUM_EPOCHS):
    arm_model.train()
    training_loss = 0.0

    for input_train_batch, target_train_batch in dataset.train_loader:
        input_train_batch, target_train_batch = input_train_batch.to(device), target_train_batch.to(device)
        optimizer.zero_grad()
        joints = arm_model(input_train_batch)
        loss = lossFunction(joints, target_train_batch)
        loss.backward()
        optimizer.step()
        training_loss += loss.item() * input_train_batch.size(0)

    arm_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for input_val_batch, target_val_batch in dataset.val_loader:
            input_val_batch, target_val_batch = input_val_batch.to(device), target_val_batch.to(device)
            joints = arm_model(input_val_batch)
            loss = lossFunction(joints, target_val_batch)
            val_loss += loss.item() * input_val_batch.size(0)

    training_loss /= len(dataset.train_loader.dataset)
    val_loss /= len(dataset.val_loader.dataset)

    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}')

    with open("arm_learning_report.csv", 'a') as file:
        file.write(str(epoch) + ',' + str(training_loss) + ',' + str(val_loss) + '\n')

    early_stopping(val_loss, arm_model)

    if early_stopping.early_stop:
        final_epoch = epoch
        print('Early stopping triggered')
        break

arm_model.save_checkpoint(final_epoch, 'arm_model_dataset_bello')
# torch.save(arm_model.state_dict(), 'saved_model.pth')
# arm_net_model.save_checkpoint(final_epoch, 'arm_model_net')
print('Training finished.')
