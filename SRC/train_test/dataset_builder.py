import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import numpy as np
from client import Client, JOINTS, DEFAULT_PORT
import math
from server import get_neutral_joint_position
from utilities.action_space import ActionSpaceArm
import torch
import time
import csv

# Set the action space for the arm
ACTION_SPACE_ARM = ActionSpaceArm()


# Function to run the main dataset generation loop
def run(cli):
    """
    Control the robotic arm to generate and record dataset by moving randomly.

    Args:
        cli (Client): Client object to interact with the server.
    """
    action = np.zeros(JOINTS)   # Initialize the action array with zeros
    action[0] = 0.3
    action[2] = math.pi
    action[3] = math.pi/3
    action[9] = 0
    action[10] = math.pi / 2

    # Open CSV file to write the dataset
    with open("dataset_file.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'joint0', 'joint3', 'joint5', 'joint7', 'paddle_y', 'paddle_z'
        ])  # Write header row

    state = cli.get_state()

    while True:
        # If paddle is under the table or too behind, reset to neutral position
        if state[13] < 0 or state[12] < -2:
            action = get_neutral_joint_position()
            cli.send_joints(action)
            time.sleep(0.2)
            action = np.zeros(JOINTS)
            cli.send_joints(action)
            time.sleep(0.2)
            action[2] = math.pi
            action[9] = 0
            action[10] = math.pi / 2
            state = cli.get_state()

        else:
            # Generate a random action within the action space range [-1, 1]
            random_action = torch.rand(ACTION_SPACE_ARM.shape[0]) * 2 - 1  # Generate a random action in range [-1, 1]
            scaled_action = ACTION_SPACE_ARM.rescale_action(random_action)

            action[0] = scaled_action[0]
            action[3] = scaled_action[1]
            action[5] = scaled_action[2]
            action[7] = scaled_action[3]

            cli.send_joints(action)  # Send the action to the server
            time.sleep(0.2)  # Wait for a short duration in which the arm take the new position
            state = cli.get_state()  # Get the new state after action

            # Prepare a line of data for the CSV file
            line = str(state[0]) + "," + str(state[3]) + "," + str(state[5]) + "," + str(state[7]) + "," + str(state[12]) + "," + str(state[13])

            # Append the data to the CSV file
            with open("dataset_file.csv", 'a') as file:
                file.write(str(line) + '\n')


def main():
    name = 'Dataset Builder'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    port = DEFAULT_PORT
    if len(sys.argv) > 2:
        port = sys.argv[2]

    host = 'localhost'
    if len(sys.argv) > 3:
        host = sys.argv[3]

    cli = Client(name, host, port)
    run(cli)


if __name__ == '__main__':
    '''
    python client_example.py name port host
    Default parameters:
     name: 'Example Client'
     port: client.DEFAULT_PORT
     host: 'localhost'

    To run the one simulation on the server, run this in 3 separate command shells:
    > python client_example.py player_A
    > python client_example.py player_B
    > python server.py

    To run a second simulation, select a different PORT on the server:
    > python client_example.py player_A 9544
    > python client_example.py player_B 9544
    > python server.py -port 9544    
    '''

    main()
