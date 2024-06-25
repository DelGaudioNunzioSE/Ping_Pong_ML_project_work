from client import Client, DEFAULT_PORT
import sys
from utilities.action_space import ActionSpaceArm
import torch
import logging
from utilities.trajectory import trajectory #, bounce_trajectory
from utilities.ddpg import DDPG
from server import get_neutral_joint_position
import numpy as np
import math
from utilities.arm_net import ArmModel

GAMMA = 0.99
TAU = 0.01
HIDDEN_SIZE_ARM = (100, 50)
ACTION_SPACE_ARM = ActionSpaceArm()
NUM_INPUTS = 2

# Create logger
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))

arm_model = ArmModel(HIDDEN_SIZE_ARM, NUM_INPUTS, ACTION_SPACE_ARM).to(device)

arm_model.load_checkpoint()

arm_model.eval()


def run(cli):


    out = False
    wait_bounce = False
    input_state = np.zeros(NUM_INPUTS)
    action = get_neutral_joint_position()
    while True:
        print("Sono qui 1")
        prev_state = cli.get_state()
        print("Sono qui 2")
        # Each state:
        # - Read the state;
        # - Set neutral (changed if the model is activated);
        # - Set cart as far ahead as possible (changed if the model is activated);
        state = cli.get_state()
        print("Sono qui 3")
        # Game finished:
        # - Reset 'out';
        if not prev_state[28] and state[28]:
            action = get_neutral_joint_position()
            action[9] = 0
            out = False
            wait_bounce = False
        elif state[21] > 0:
            action = get_neutral_joint_position()
            action[0] = 0.3
            action[9] = 0

        # Activation:
        # - If the ball is coming to us (negative ball-y-velocity);
        # - And the game is playing;
        # - And the ball (cat) is on the table (negative ball-z-position)
        if state[21] < 0 and state[28] and not out and state[19] > 0:
            # Calculate the trajectory to check if the ball go on our side of the table
            z = 0.1
            x, y = trajectory(state, z)
            print("x: ", x, "y: ", y)
            if x is not None and y is not None:
                if (prev_state[21] * state[21]) < 0 and ((x < -0.7 or x > 0.7) or (y < -0.1 or y > 1.3)):
                    print("va fuori")
                    action = get_neutral_joint_position()
                    out = True
                    if x <= 0:
                        action[1] = 0.8
                    else:
                        action[1] = -0.8
                else:
                    print("Sono arrivato qui con y = ", y)
                    if y >= 0.3:
                        wait_bounce = True
                    else:
                        input_state[0] = y + 0.3
                        input_state[1] = 0.5

                        input_state = torch.Tensor(input_state)

                        arm_action = arm_model(input_state)
                        action[0] = arm_action[0]
                        action[3] = arm_action[1]
                        action[5] = arm_action[2]
                        action[7] = arm_action[3] - math.pi / 6
                        action[9] = math.pi * 3 / 4

                    # smash = False
                    # dont_smash = False
                    # z_smash = 0.4
                    # while not smash and not dont_smash:
                    #     x_smash, y_smash = trajectory(state, z_smash)
                    #     if x_smash is not None and y_smash is not None:
                    #         if y_smash <= 0.1:
                    #             smash = True
                    #             y = y_smash
                    #             x = x_smash
                    #             z = z_smash
                    #     if z_smash >= 0.8:
                    #         dont_smash = True
                    #     else:
                    #         z_smash += 0.1

                    action[1] = x

                    # input_state = np.zeros(NUM_INPUTS)
                    # input_state[0] = y
                    # input_state[1] = z
                    #
                    # input_state = torch.Tensor(input_state)
                    #
                    # arm_action = arm_model(input_state)
                    # action[0] = arm_action[0]
                    # action[3] = arm_action[1]
                    # action[5] = arm_action[2]
                    # action[7] = arm_action[3]


                    # y += 0.35
                    # y = max(-0.3, min(y, 0.3))
                    # y = 0.3
                    # x = max(-0.8, min(x, 0.8))

                # action[0] = y
                # action[1] = x
            # else:
                # x, y = trajectory(state, 0)

            # input_state = np.zeros(NUM_INPUTS)
            # input_state[0] = state[18]
            # input_state[1] = state[19]

            # input_state[0] = y
            # input_state[1] = z + 0.1
            #
            # input_state = torch.Tensor(input_state)
            #
            # arm_action = arm_model(input_state)
            # action[0] = arm_action[0]
            # action[3] = arm_action[1]
            # action[5] = arm_action[2]
            # action[7] = arm_action[3]

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
