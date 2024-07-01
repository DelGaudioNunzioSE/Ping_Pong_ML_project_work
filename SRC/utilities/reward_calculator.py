"""

    Machine Learning Project Work: Tennis Table Tournament
    Group 2:
        Ciaravola Giosuè - g.ciaravola3@studenti.unisa.it
        Conato Christian - c.conato@studenti.unisa.it
        Del Gaudio Nunzio - n.delgaudio5@studenti.unisa.it
        Garofalo Mariachiara - m.garofalo38@studenti.unisa.it

    ---------------------------------------------------------------

    reward_calculator.py

    File containing the reward calculation to use during
    reinforcement training.

"""

import numpy as np
from utilities.trajectory import trajectory


def calculate_paddle_reward(prev_state, state, point_state):
    """
    Calculate the reward for the paddle based on the current and previous states.

    Args:
        prev_state (np.array): Previous state of the environment.
        state (np.array): Current state of the environment.
        point_state (np.array or None): State representing a point event (e.g., scoring point).

    Returns:
        float: Calculated reward for the paddle.
    """

    # Ball has changed direction in y-axis
    if prev_state[21] * state[21] < 0:
        x, y = trajectory(state, 0)  # Calculate trajectory to reach z = 0
        if x is not None and y is not None:
            if -0.7 < x < 0.7 and 1.2 < y < 2.4:

                x = abs(x)

                reward = (x * 20) + 10  # Calculate reward based on ball's x position

                if reward < 15:
                    reward = 15  # Ensure minimum reward

            else:
                # Ball is out of opponent field
                reward = -10
        else:
            reward = 0  # Invalid trajectory calculation
    # If the episode is ended without catch the ball
    else:
        reward = -15

    # Point state is available (e.g., scoring point conditions)
    if point_state is not None:

        # Reward for scoring a point (if the reward is not already positive)
        if point_state[34] > prev_state[34] and reward <= 0:
            reward = 20

        # Reward for losing a point (if we don't miss the ball)
        if point_state[35] > prev_state[35] and reward != -15:
            reward = -10

    return reward
