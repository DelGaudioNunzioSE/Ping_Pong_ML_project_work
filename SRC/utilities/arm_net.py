import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import logging
import os
import gc


# Configure the logger for the ArmNet module
logger = logging.getLogger('ddpg')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


# Custom initialization function
def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)  # Calculate fan_in if not specified

    w = 1. / np.sqrt(fan_in)  # Calculate the limit for uniform initialization
    nn.init.uniform_(tensor, -w, w)  # Initialize the tensor uniformly


# Definition of the Actor network
class ArmModel(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(ArmModel, self).__init__()
        self.action_space = action_space  # Action space
        num_outputs = action_space.shape[0]  # Number of actions

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])  # First linear layer
        self.ln1 = nn.LayerNorm(hidden_size[0])  # LayerNorm normalization

        # Layer 2
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])  # Second linear layer
        self.ln2 = nn.LayerNorm(hidden_size[1])  # LayerNorm normalization

        # Layer 3
        self.linear3 = nn.Linear(hidden_size[1], hidden_size[2])  # Second linear layer
        self.ln3 = nn.LayerNorm(hidden_size[2])  # LayerNorm normalization

        # Output Layer
        self.mu = nn.Linear(hidden_size[2], num_outputs)  # Output layer that produces actions

        # Weight Initialization
        fan_in_uniform_init(self.linear1.weight)  # Initialize weights of the first layer
        fan_in_uniform_init(self.linear1.bias)  # Initialize biases of the first layer

        fan_in_uniform_init(self.linear2.weight)  # Initialize weights of the second layer
        fan_in_uniform_init(self.linear2.bias)  # Initialize biases of the second layer

        nn.init.uniform_(self.mu.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)  # Initialize weights of the output layer
        nn.init.uniform_(self.mu.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)  # Initialize biases of the output layer

    def forward(self, inputs):
        x = inputs.to(device)  # Move the input to the correct device

        # Layer 1
        x = self.linear1(x)  # Pass through the first linear layer
        x = self.ln1(x)  # Normalize
        x = F.relu(x)  # ReLU activation function

        # Layer 2
        x = self.linear2(x)  # Pass through the second linear layer
        x = self.ln2(x)  # Normalize
        x = F.relu(x)  # ReLU activation function

        # Layer 3
        x = self.linear3(x)  # Pass through the first linear layer
        x = self.ln3(x)  # Normalize
        x = F.relu(x)  # ReLU activation function

        # Output
        mu = self.mu(x)

        return mu


# class ArmNET(object):
#
#     def __init__(self, hidden_size, num_inputs, action_space, checkpoint_dir=None):
#
#         self.action_space = action_space
#
#         # Define the actor network and its target network
#         self.model = ArmModel(hidden_size, num_inputs, self.action_space).to(device)
#
#         # Define the optimizers for both networks
#         self.optimizer = Adam(self.model.parameters(), lr=0.001)  # optimizer for the actor network
#
#         # Set the directory to save the models
#         if checkpoint_dir is None:
#             self.checkpoint_dir = "../saved_models_arm/"
#         else:
#             self.checkpoint_dir = checkpoint_dir
#         os.makedirs(self.checkpoint_dir, exist_ok=True)
#         logger.info('Saving all checkpoints to {}'.format(self.checkpoint_dir))
#
#     def calc_joints(self, state):
#
#         x = state.to(device, dtype=torch.float32)
#
#         # Get the continuous action value to perform in the env
#         #self.model.eval()  # Sets the actor in evaluation mode
#         mu = self.model(x)
#         #self.model.train()  # Sets the actor in training mode
#         mu = mu.data
#
#         # Rescale mu to be within the joint ranges
#         mu = self.action_space.clamp_action(mu)
#
#         return mu
#
#     def save_checkpoint(self, last_epoch, model_name):
#
#         # Construct the checkpoint file name using the checkpoint directory and the current timestep
#         checkpoint_name = self.checkpoint_dir + '/{}_epoch_{}.pth.tar'.format(model_name, last_epoch)
#
#         # Log a message indicating that a checkpoint is being saved
#         logger.info('Saving checkpoint...')
#
#         # Create a dictionary that contains all the elements to be saved
#         checkpoint = {
#             'last_epoch': last_epoch,  # Save the last epoch
#             'model': self.model.state_dict(),  # Save the weights and biases of the actor network
#             'optimizer': self.optimizer.state_dict(),  # Save the state of the actor optimizer
#         }
#
#         # Log a message indicating that the model is being saved
#         logger.info('Saving model at epoch {}...'.format(last_epoch))
#
#         # Save the checkpoint dictionary to the specified file
#         torch.save(checkpoint, checkpoint_name)
#
#         # Perform garbage collection to free up unused memory
#         gc.collect()
#
#         # Log a message indicating that the model has been successfully saved
#         logger.info('Saved model at epoch {} to {}'.format(last_epoch, self.checkpoint_dir))
#
#     def get_path_of_latest_file(self):
#         """
#         Returns the latest created file in 'checkpoint_dir'
#         """
#         files = [file for file in os.listdir(self.checkpoint_dir) if (file.endswith(".pt") or file.endswith(".tar"))]
#         filepaths = [os.path.join(self.checkpoint_dir, file) for file in files]
#         last_file = max(filepaths, key=os.path.getctime)
#         return os.path.abspath(last_file)
#
#     def load_checkpoint(self, checkpoint_path=None):
#
#         if checkpoint_path is None:
#             checkpoint_path = self.get_path_of_latest_file()
#
#         if os.path.isfile(checkpoint_path):
#             logger.info("Loading checkpoint...({})".format(checkpoint_path))
#             key = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#             checkpoint = torch.load(checkpoint_path, map_location=key)
#             start_epoch = checkpoint['last_epoch'] + 1
#             self.model.load_state_dict(checkpoint['model'])
#             self.optimizer.load_state_dict(checkpoint['optimizer'])
#
#             gc.collect()
#
#             logger.info('Loaded model at epoch {} from {}'.format(start_epoch, checkpoint_path))
#             return start_epoch
#         else:
#             raise OSError('Checkpoint not found')
#
#     def set_eval(self):
#         self.model.eval()
#
#     def set_train(self):
#         self.model.train()
