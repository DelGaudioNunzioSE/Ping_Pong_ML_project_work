"""

    Machine Learning Project Work: Tennis Table Tournament
    Group 2:
        Ciaravola Giosuè - g.ciaravola3@studenti.unisa.it
        Conato Christian - c.conato@studenti.unisa.it
        Del Gaudio Nunzio - n.delgaudio5@studenti.unisa.it
        Garofalo Mariachiara - m.garofalo38@studenti.unisa.it

    ---------------------------------------------------------------

    replay_memory.py

    File containing the class representing the replay buffer where
    to record transitions during reinforcement learning.

"""

import random
from collections import namedtuple

# Define a named tuple for Transition
Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward')
                        )


class ReplayMemory(object):
    """
    Replay memory for storing and sampling transitions.
    """

    def __init__(self, capacity):
        """
        Initialize ReplayMemory.

        Args:
            capacity (int): Capacity of the replay memory.
        """
        self.capacity = capacity
        self.memory = []  # Initialize an empty list for storing transitions
        self.position = 0  # Initialize the starting position in the memory

    def push(self, *args):
        """
        Saves a transition.

        Args:
            *args: Elements of a transition (state, action, done, next_state, reward).
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # Extend the memory if not yet at capacity
        self.memory[self.position] = Transition(*args)  # Store the transition at the current position
        self.position = (self.position + 1) % self.capacity  # Update the position for the next insertion

    def sample(self, batch_size):
        """
        Samples a batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            List[Transition]: Sampled transitions.
        """
        return random.sample(self.memory, batch_size)  # Randomly sample a batch of transitions

    def __len__(self):
        """
        Returns the current length of the memory.

        Returns:
            int: Current length of the memory.
        """
        return len(self.memory)  # Return the current length of the memory
