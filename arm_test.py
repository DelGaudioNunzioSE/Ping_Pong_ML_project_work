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

GAMMA = 0.99
TAU = 0.01
HIDDEN_SIZE_ARM = (400, 300)
ACTION_SPACE_ARM = ActionSpaceArm()
INPUT_SPACE_ARM = 8

# Create logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))

arm_agent = DDPG(GAMMA,
                 TAU,
                 HIDDEN_SIZE_ARM,
                 INPUT_SPACE_ARM,
                 ACTION_SPACE_ARM
                 )

arm_agent.load_checkpoint()

arm_agent.set_eval()


def run(cli):
    prev_state = cli.get_state()
    action = get_neutral_joint_position()
    action[9] = 0
    out = False
    while True:
        state = cli.get_state()

        # Game finished
        if not prev_state[28] and state[28]:
            action = get_neutral_joint_position()
            action[7] = math.pi * 1 / 2
            action[9] = -math.pi * 3 / 4
            out = False

        if state[21] < 0 and state[28] and not out and state[19] > 0:
            x, y = trajectory(state)
            if x is not None and y is not None:
                if (prev_state[21] * state[21]) < 0 and ((x < -0.8 or x > 0.8) or (y < -0.1 or y > 1.3)):
                    print("va fuori")
                    action = get_neutral_joint_position()
                    out = True
                    if x <= 0:
                        x = 0.8
                    else:
                        x = -0.8
                else:
                    y += 0.35
                    y = max(-0.3, min(y, 0.3))
                    x = max(-0.8, min(x, 0.8))

                action[0] = y
                action[1] = x

            input_state = np.zeros(INPUT_SPACE_ARM)
            input_state[0] = state[3]
            input_state[1] = state[5]
            input_state[2] = state[11]
            input_state[3] = state[12]
            input_state[4] = state[13]
            input_state[5] = state[17]
            input_state[6] = state[18]
            input_state[7] = state[19]

            input_state = torch.Tensor(input_state)

            arm_action = arm_agent.calc_action(input_state)
            action[3] = arm_action[0]
            action[5] = arm_action[1]

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
