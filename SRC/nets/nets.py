"""

    Machine Learning Project Work: Tennis Table Tournament
    Group 2:
        Ciaravola Giosuè - g.ciaravola3#studenti.unisa.it
        Conato Christian - c.conato@studenti.unisa.it
        Del Gaudio Nunzio - n.delgaudio5@studenti.unisa.it
        Garofalo Mariachiara - m.garofalo38@studenti.unisa.it

    ---------------------------------------------------------------

    nets.py

    File containing the model structures used as Actor and Critic
    in the DDPG, trained via reinforcement learning.

"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for the final initialization of weights and biases
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


# Custom initialization function
def fan_in_uniform_init(tensor, fan_in=None):
    """
    Utility function for initializing weights and biases of the network.

    Parameters:
    tensor (torch.Tensor): The tensor to be initialized.
    fan_in (int, optional): The number of input units in the weight tensor.
    """
    if fan_in is None:
        fan_in = tensor.size(-1)  # Calculate fan_in if not specified

    w = 1. / np.sqrt(fan_in)  # Calculate the limit for uniform initialization
    nn.init.uniform_(tensor, -w, w)  # Initialize the tensor uniformly


# Definition of the Actor network
class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        """
        Initialize the Actor network.

        Parameters:
        hidden_size (list): List of integers specifying the number of units in hidden layers.
        num_inputs (int): Number of input features.
        action_space (utilities.action_space): Action space specification.
        """
        super(Actor, self).__init__()
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

        fan_in_uniform_init(self.linear3.weight)  # Initialize weights of the third layer
        fan_in_uniform_init(self.linear3.bias)  # Initialize biases of the third layer

        nn.init.uniform_(self.mu.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)  # Initialize weights of the output layer
        nn.init.uniform_(self.mu.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)  # Initialize biases of the output layer

    def forward(self, inputs):
        """
        Perform the forward pass through the Actor network.

        Parameters:
        inputs (torch.Tensor): Input tensor to the network.

        Returns:
        torch.Tensor: Output tensor after passing through the network.
        """
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
        x = self.linear3(x)  # Pass through the third linear layer
        x = self.ln3(x)  # Normalize
        x = F.relu(x)  # ReLU activation function

        # Output
        mu = torch.tanh(self.mu(x))  # Tanh activation function to limit the actions
        return mu


# Definition of the Critic network
class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        """
        Initialize the Critic network.

        Parameters:
        hidden_size (list): List of integers specifying the number of units in hidden layers.
        num_inputs (int): Number of input features.
        action_space (utilities.action_space): Action space specification.
        """
        super(Critic, self).__init__()
        self.action_space = action_space  # Action space
        num_outputs = action_space.shape[0]  # Number of actions

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])  # First linear layer
        self.ln1 = nn.LayerNorm(hidden_size[0])  # LayerNorm normalization

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])  # Second linear layer with state and actions as input
        self.ln2 = nn.LayerNorm(hidden_size[1])  # LayerNorm normalization

        # Layer 3
        self.linear3 = nn.Linear(hidden_size[1], hidden_size[2])  # Third linear layer
        self.ln3 = nn.LayerNorm(hidden_size[2])  # LayerNorm normalization

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[2], 1)  # Output layer that produces the value

        # Weight Initialization
        fan_in_uniform_init(self.linear1.weight)  # Initialize weights of the first layer
        fan_in_uniform_init(self.linear1.bias)  # Initialize biases of the first layer

        fan_in_uniform_init(self.linear2.weight)  # Initialize weights of the second layer
        fan_in_uniform_init(self.linear2.bias)  # Initialize biases of the second layer

        fan_in_uniform_init(self.linear3.weight)  # Initialize weights of the third layer
        fan_in_uniform_init(self.linear3.bias)  # Initialize biases of the third layer

        nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)  # Initialize weights of the output layer
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)  # Initialize biases of the output layer

    def forward(self, inputs, actions):
        """
        Perform the forward pass through the Critic network.

        Parameters:
        inputs (torch.Tensor): Input tensor (states) to the network.
        actions (torch.Tensor): Action tensor to the network.

        Returns:
        torch.Tensor: Output tensor (value) after passing through the network.
        """
        x = inputs.to(device)  # Move the input to the correct device
        actions = actions.to(device)  # Move the actions to the correct device

        # Layer 1
        x = self.linear1(x)  # Pass through the first linear layer
        x = self.ln1(x)  # Normalize
        x = F.relu(x)  # ReLU activation function

        # Layer 2
        x = torch.cat((x, actions), 0)  # Concatenate the state and actions
        x = self.linear2(x)  # Pass through the second linear layer
        x = self.ln2(x)  # Normalize
        x = F.relu(x)  # ReLU activation function

        # Layer 3
        x = self.linear3(x)  # Pass through the third linear layer
        x = self.ln3(x)  # Normalize
        x = F.relu(x)  # ReLU activation function

        # Output
        V = self.V(x)  # Calculate the value
        return V
