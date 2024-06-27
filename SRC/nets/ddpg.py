import gc
import logging
import os
import torch

import torch.nn.functional as F
from torch.optim import Adam

from nets.nets import Actor, Critic

# Configure the logger for the DDPG module
logger = logging.getLogger('ddpg')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Soft update function to update the target networks' parameters
def soft_update(target, source, tau):
    """
    Perform soft update of target network parameters with source network parameters.

    Args:
        target (nn.Module): Target network to update.
        source (nn.Module): Source network to copy from.
        tau (float): Interpolation parameter (soft update rate).
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# Hard update function to copy parameters from source to target networks
def hard_update(target, source):
    """
    Perform hard update (copy) of target network parameters from source network parameters.

    Args:
        target (nn.Module): Target network to update.
        source (nn.Module): Source network to copy from.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG(object):

    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, checkpoint_dir=None):
        """
        Deep Deterministic Policy Gradient (DDPG) class for reinforcement learning.

        Args:
            gamma (float): Discount factor for future rewards.
            tau (float): Soft update rate for target networks.
            hidden_size (list): List of integers representing sizes of hidden layers for networks.
            num_inputs (int): Size of the input state space.
            action_space (object): Object defining the action space of the environment.
            checkpoint_dir (str, optional): Directory path to save model checkpoints. Defaults to None.
        """

        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space

        # Define the actor network and its target network
        self.actor = Actor(hidden_size, num_inputs, self.action_space).to(device)
        self.actor_target = Actor(hidden_size, num_inputs, self.action_space).to(device)

        # Define the critic network and its target network
        self.critic = Critic(hidden_size, num_inputs, self.action_space).to(device)
        self.critic_target = Critic(hidden_size, num_inputs, self.action_space).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-4)  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)  # optimizer for the critic network

        # Ensure both target networks start with the same weights as the main networks
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Set the directory to save the models
        if checkpoint_dir is None:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Define the relative path to the directory where you want to save the models
            self.checkpoint_dir = os.path.join(current_dir, "saved_models")
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Define the relative path to the directory where you want to save the models
            self.checkpoint_dir = os.path.join(current_dir, checkpoint_dir)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info('Saving all checkpoints to {}'.format(self.checkpoint_dir))

    def calc_action(self, state, action_noise=None):
        """
        Calculate action to take in a given state, optionally add noise for exploration.

        Args:
            state (torch.Tensor): Input state to evaluate the action on.
            action_noise (NoiseGenerator, optional): Noise generator object for exploration. Defaults to None.

        Returns:
            torch.Tensor: Evaluated action for the given state.
        """

        x = state.to(device)

        # Get the continuous action value to perform in the env
        self.actor.eval()  # Sets the actor in evaluation mode
        mu = self.actor(x)
        self.actor.train()  # Sets the actor in training mode
        mu = mu.data

        # During training, we add noise for exploration (if specified)
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(device)
            mu += noise
            # Clamp the output to ensure it remains between -1 and 1
            mu = mu.clamp(-1, 1)

        # Rescale mu to be within the joint ranges
        mu = self.action_space.rescale_action(mu)

        return mu

    def update_params(self, batch):
        """
        Updates the parameters/networks of the agent according to the given batch.
        This means we ...
            1. Compute the targets
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update

        Arguments:
            batch:  Batch to perform the training of the parameters
        """

        # Get tensors from the batch
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.cat(batch.done).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_checkpoint(self, last_timestep, model_name, replay_buffer=None):
        """
        Save networks' parameters and other training parameters to a file in 'checkpoint_dir'.

        Args:
            last_timestep (int): Last timestep before saving the checkpoint.
            model_name (str): Name of the model checkpoint.
            replay_buffer (ReplayBuffer, optional): Current replay buffer object. Defaults to None.
        """

        # Construct the checkpoint file name using the checkpoint directory and the current timestep
        checkpoint_name = self.checkpoint_dir + '/{}_ep_{}.pth.tar'.format(model_name, last_timestep)

        # Log a message indicating that a checkpoint is being saved
        logger.info('Saving checkpoint...')

        # Create a dictionary that contains all the elements to be saved
        checkpoint = {
            'last_timestep': last_timestep,  # Save the last timestep
            'actor': self.actor.state_dict(),  # Save the weights and biases of the actor network
            'critic': self.critic.state_dict(),  # Save the weights and biases of the critic network
            'actor_target': self.actor_target.state_dict(),  # Save the weights and biases of the actor target network
            'critic_target': self.critic_target.state_dict(),  # Save the weights and biases of the critic target network
            'actor_optimizer': self.actor_optimizer.state_dict(),  # Save the state of the actor optimizer
            'critic_optimizer': self.critic_optimizer.state_dict(),  # Save the state of the critic optimizer
            'replay_buffer': replay_buffer,  # Save the current state of the replay buffer
        }

        # Log a message indicating that the model is being saved
        logger.info('Saving model at timestep {}...'.format(last_timestep))

        # Save the checkpoint dictionary to the specified file
        torch.save(checkpoint, checkpoint_name)

        # Perform garbage collection to free up unused memory
        gc.collect()

        # Log a message indicating that the model has been successfully saved
        logger.info('Saved model at timestep {} to {}'.format(last_timestep, self.checkpoint_dir))

    def get_path_of_latest_file(self):
        """
        Returns the latest created file in 'checkpoint_dir'
        """
        files = [file for file in os.listdir(self.checkpoint_dir) if (file.endswith(".pt") or file.endswith(".tar"))]
        filepaths = [os.path.join(self.checkpoint_dir, file) for file in files]
        last_file = max(filepaths, key=os.path.getctime)
        return os.path.abspath(last_file)

    def load_checkpoint(self, checkpoint_path=None):
        """
        Saving the networks and all parameters from a given path. If the given path is None
        then the latest saved file in 'checkpoint_dir' will be used.

        Arguments:
            checkpoint_path:    File to load the model from

        """

        if checkpoint_path is None:
            checkpoint_path = self.get_path_of_latest_file()

        if os.path.isfile(checkpoint_path):
            logger.info("Loading checkpoint...({})".format(checkpoint_path))
            key = 'cuda' if torch.cuda.is_available() else 'cpu'

            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            replay_buffer = checkpoint['replay_buffer']

            gc.collect()
            logger.info('Loaded model at timestep {} from {}'.format(start_timestep, checkpoint_path))
            return start_timestep, replay_buffer
        else:
            raise OSError('Checkpoint not found')

    def set_eval(self):
        """
        Sets the model in evaluation mode

        """
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        """
        Sets the model in training mode

        """
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def get_network(self, name):
        """
        Returns the specified network (either Actor or Critic)

        Arguments:
            name: Name of the network ('Actor' or 'Critic')
        """
        if name == 'Actor':
            return self.actor
        elif name == 'Critic':
            return self.critic
        else:
            raise NameError('name \'{}\' is not defined as a network'.format(name))
