from client import Client, DEFAULT_PORT
import sys
from utilities.action_space import ActionSpaceArm
import torch
import logging
from utilities.trajectory import trajectory
from utilities.ddpg import DDPG
from server import get_neutral_joint_position
import numpy as np
import math
from utilities.arm_net import ArmModel

GAMMA = 0.99
TAU = 0.01
HIDDEN_SIZE_ARM = (400, 300, 200)
ACTION_SPACE_ARM = ActionSpaceArm()
NUM_INPUTS = 2

# Create logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))

arm_net_model = ArmModel(HIDDEN_SIZE_ARM, NUM_INPUTS, ACTION_SPACE_ARM).to(device)

arm_net_model.load_state_dict(torch.load('saved_model.pth'))

arm_net_model.eval()


def run(cli):
    prev_state = cli.get_state()
    action = get_neutral_joint_position()
    action[9] = 0
    out = False
    z = 0
    while True:
        state = cli.get_state()

        # Game finished
        if not prev_state[28] and state[28]:
            action = get_neutral_joint_position()
            action[9] = 0

            out = False

        if state[21] < 0 and state[28] and not out and state[19] > 0:
            x, y = trajectory(state, z)
            if x is not None and y is not None:
                if (prev_state[21] * state[21]) < 0 and ((x < -0.8 or x > 0.8) or (y < -0.2 or y > 1.3)):
                    print("va fuori")
                    action = get_neutral_joint_position()
                    out = True
                    if x <= 0:
                        x = 0.8
                    else:
                        x = -0.8
                else:
                    # y += 0.35
                    # y = max(-0.3, min(y, 0.3))
                    # y = 0.3
                    x = max(-0.8, min(x, 0.8))

                # action[0] = y
                action[1] = x

            input_state = np.zeros(NUM_INPUTS)
            # input_state[0] = state[18]
            # input_state[1] = state[19]

            input_state[0] = y
            input_state[1] = z + 0.1

            input_state = torch.Tensor(input_state)

            arm_action = arm_net_model(input_state)
            action[0] = arm_action[0]
            action[3] = arm_action[1]
            action[5] = arm_action[2]
            action[7] = arm_action[3]

        cli.send_joints(action)

        prev_state = state


def main():
    name = 'ARM test'
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
    python ddpg_train.py name port host
    Default parameters:
     name: 'Example Client'
     port: client.DEFAULT_PORT
     host: 'localhost'

    To run the one simulation on the server, run this in 3 separate command shells:
    > python ddpg_train.py player_A
    > python ddpg_train.py player_B
    > python server.py

    To run a second simulation, select a different PORT on the server:
    > python ddpg_train.py player_A 9544
    > python ddpg_train.py player_B 9544
    > python server.py -port 9544    
    '''

    main()
