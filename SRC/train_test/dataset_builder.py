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

ACTION_SPACE_ARM = ActionSpaceArm()
DATASET_SIZE = 1000


def run(cli):
    # action = get_neutral_joint_position()
    action = np.zeros(JOINTS)
    action[0] = 0.3
    action[2] = math.pi
    action[3] = math.pi/3
    action[9] = 0
    action[10] = math.pi / 2

    with open("dataset_file.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'joint0', 'joint3', 'joint5', 'joint7', 'paddle_y', 'paddle_z'
        ])

    state = cli.get_state()
    registered = 0

    while registered < DATASET_SIZE:
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

            random_action = torch.rand(ACTION_SPACE_ARM.shape[0]) * 2 - 1  # Generate a random action in range [-1, 1]
            scaled_action = ACTION_SPACE_ARM.rescale_action(random_action)

            action[0] = scaled_action[0]
            action[3] = scaled_action[1]
            action[5] = scaled_action[2]
            action[7] = scaled_action[3]

            cli.send_joints(action)
            time.sleep(0.2)
            state = cli.get_state()

            line = str(state[0]) + "," + str(state[3]) + "," + str(state[5]) + "," + str(state[7]) + "," + str(state[12]) + "," + str(state[13])

            with open("dataset_file.csv", 'a') as file:
                file.write(str(line) + '\n')

            registered += 1


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
