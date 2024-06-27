import numpy as np
from utilities.trajectory import trajectory


def calculate_paddle_reward(prev_state, state, point_state, done):

    paddle_pos = np.array(state[11:14])
    ball_pos = np.array(state[17:20])

    distance = np.linalg.norm(paddle_pos - ball_pos)

    if done:
        if prev_state[21] * state[21] < 0:
            x, y = trajectory(state, 0)
            if x is not None and y is not None:

                # print("x: ", x, "y: ", y, "vy: ", state[21])
                if -0.7 < x < 0.7 and 1.2 < y < 2.4:

                    x = abs(x)

                    reward = (x * 20) + 10

                else:
                    reward = -5
            else:
                reward = 0
        elif distance > 0.3:
            reward = -10
        else:
            reward = 0
    else:
        reward = 0

    if point_state is not None:

        if point_state[34] > prev_state[34] and reward <= 0:
            reward = 20

        if point_state[35] > prev_state[35] and reward != -10:
            reward = -5

    return reward


def calculate_arm_reward(prev_state, next_state):
    # Extract paddle and ball positions from the states
    prev_paddle_pos = np.array(prev_state[11:14])
    prev_ball_pos = np.array(prev_state[17:20])

    next_paddle_pos = np.array(next_state[11:14])
    next_ball_pos = np.array(next_state[17:20])

    # Calculate the Euclidean distance between the ball and the center of the paddle
    prev_distance = np.linalg.norm(prev_paddle_pos - prev_ball_pos)
    next_distance = np.linalg.norm(next_paddle_pos - next_ball_pos)

    reward = prev_distance - next_distance

    if reward < 0:
        reward = 0

    if next_paddle_pos[2] < 0.5:
        reward = -2

    if next_paddle_pos[1] < -0.7:
        reward = -3

    """
    z = (next_paddle_pos[2] - 0.3)
    if reward < 0:
        if z < 0:
            reward = reward * (-z)
        else:
            reward = reward * z
    else:
        reward = reward * z
        """


    # Check if the paddle is below the table (z < 0)
    if next_paddle_pos[2] < 0:
        reward = -5  # Assign a very negative reward

    return reward


def calculate_reward(prev_state, next_state):
    """
    Compute the reward based on the previous and current state.

    Arguments:
        prev_state: The state vector before the current action.
        next_state: The current state vector after the action.

    Returns:
        reward: The computed reward value.
    """
    # Extract relevant information from the state vectors
    prev_score = prev_state[34]
    prev_opponent_score = prev_state[35]
    prev_ball_touched_robot = prev_state[31]

    curr_score = next_state[34]
    curr_opponent_score = next_state[35]
    curr_ball_touched_robot = next_state[31]
    curr_ball_in_opponent_half = next_state[32]

    reward = 0

    # Reward for scoring a point
    if curr_score > prev_score:
        reward += 10

    # Penalty for opponent scoring a point
    if curr_opponent_score > prev_opponent_score:
        reward -= 10

    # Reward for hitting the ball
    if curr_ball_touched_robot and not prev_ball_touched_robot:
        reward += 1

    # Reward for keeping the ball in the opponent's half
    if curr_ball_in_opponent_half:
        reward += 0.1

    # Calculate the distance between the paddle center and the ball
    paddle_center = next_state[11:14]
    ball_position = next_state[17:20]
    distance = np.linalg.norm(paddle_center - ball_position)

    # Reward inversely proportional to the distance (smaller distance, higher reward)
    reward += max(0, 1 - distance)  # Ensure non-negative reward contribution

    return reward

