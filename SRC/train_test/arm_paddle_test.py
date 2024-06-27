import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import torch
from client import Client, DEFAULT_PORT
import sys
from utilities.action_space import ActionSpaceArm, ActionSpacePaddleSmash, ActionSpacePaddleDontWait
from server import get_neutral_joint_position
import numpy as np
from nets.arm_net import ArmModel
from utilities.trajectory import trajectory, max_height_point
from nets.ddpg import DDPG

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: ", device)


GAMMA = 0.99
TAU = 0.01
HIDDEN_SIZE_PADDLE = (300, 200, 100)
ACTION_SPACE_SMASH = ActionSpacePaddleSmash()
ACTION_SPACE_DONT_WAIT = ActionSpacePaddleDontWait()
NUM_INPUTS_PADDLE = 6

smash_agent = DDPG(GAMMA,
                 TAU,
                 HIDDEN_SIZE_PADDLE,
                 NUM_INPUTS_PADDLE,
                 ACTION_SPACE_SMASH,
                 checkpoint_dir="saved_models_smash"
                 )

smash_agent.load_checkpoint()
smash_agent.set_eval()

dont_wait_agent = DDPG(GAMMA,
                 TAU,
                 HIDDEN_SIZE_PADDLE,
                 NUM_INPUTS_PADDLE,
                 ACTION_SPACE_DONT_WAIT,
                 checkpoint_dir="saved_models_dont_wait"
                 )

dont_wait_agent.load_checkpoint()
dont_wait_agent.set_eval()

HIDDEN_SIZE_ARM = (100, 50)
ACTION_SPACE_ARM = ActionSpaceArm()
NUM_INPUTS_ARM = 2

arm_model = ArmModel(HIDDEN_SIZE_ARM, NUM_INPUTS_ARM, ACTION_SPACE_ARM).to(device)

arm_model.load_checkpoint()
arm_model.eval()


def run(cli):

    out = False
    wait_bounce_to_smash = False
    stance_chosen = False

    input_state_arm = np.zeros(NUM_INPUTS_ARM)
    input_state_paddle = np.zeros(NUM_INPUTS_PADDLE)

    action = get_neutral_joint_position()
    prev_state = cli.get_state()

    while True:

        # Each state:
        # - Read the state;
        # - Set neutral (changed if the model is activated);
        # - Set cart as far ahead as possible (changed if the model is activated);

        state = cli.get_state()

        # if state[22] < 0:
        #     print("vy: ", state[21])

        # Game finished:
        # - Reset 'out';
        if (not prev_state[28] and state[28]) or (state[21] > 0 and prev_state[18] > 1.2):

            # print("START GAME OR CHOICE!")

            x = 0
            y = 0.8
            z = 0.35

            input_state_arm[0] = y
            input_state_arm[1] = z

            input_state_arm = torch.Tensor(input_state_arm)

            arm_action = arm_model(input_state_arm)
            action[0] = arm_action[0]
            action[1] = x
            action[3] = arm_action[1]
            action[5] = arm_action[2]
            action[7] = arm_action[3]
            action[9] = 0.65

            out = False
            wait_bounce_to_smash = False
            stance_chosen = False

        # Activation:
        # - If the ball is coming to us (negative ball-y-velocity);
        # - And the game is playing;
        # - And the ball (cat) is on the table (negative ball-z-position)
        if state[21] < 0 and state[28] and not out and state[19] > 0:
            # Check if touch the net
            if (prev_state[21] * state[21]) <= 0:
                # Calculate the trajectory to check if the ball go on our side of the table
                x, y = trajectory(state)
                if x is not None and y is not None:
                    if (x < -0.75 or x > 0.75) or (y < -0.2 or y > 1.2):
                        print("va fuori")
                        action = get_neutral_joint_position()
                        out = True
                        if x <= 0:
                            action[1] = 0.8
                        else:
                            action[1] = -0.8
                    else:
                        x_max, y_max, z_max = max_height_point(state)
                        # print("z max: ", z_max, "y :", y, "vz:", state[22])
                        if state[22] > 0 and y > 0.2 and z_max is not None and z_max >= 0.75:
                            wait_bounce_to_smash = True
                            action = get_neutral_joint_position()
                            # print("sono dentro")
                        else:
                            wait_bounce_to_smash = False

                    if not out and not wait_bounce_to_smash and not stance_chosen:

                        print("DON'T WAIT!")

                        stance_chosen = True
                        input_state_arm[0] = y + 0.2
                        input_state_arm[1] = 0.6

                        input_state_arm = torch.Tensor(input_state_arm)

                        arm_action = arm_model(input_state_arm)
                        action[0] = arm_action[0]
                        action[1] = x
                        action[3] = arm_action[1]
                        action[5] = arm_action[2]
                        action[7] = arm_action[3]
                        action[9] = 1.5

            # print("z prima ", prev_state[22], "z dopo", state[22])
            if not out and wait_bounce_to_smash and prev_state[22] < 0 and state[22] > 0 and prev_state[18] < 1.2 and not stance_chosen:

                z_smash = 1.4
                while not stance_chosen:
                    x_smash, y_smash = trajectory(state, z_smash)
                    if x_smash is not None and y_smash is not None:
                        if y_smash <= 0.1:
                            stance_chosen = True
                            y = y_smash + 0.2
                            x = x_smash
                            z = z_smash - 0.3
                    else:
                        z_smash -= 0.1

                print("SMASH!")

                input_state_arm[0] = y
                input_state_arm[1] = z

                input_state_arm = torch.Tensor(input_state_arm)

                arm_action = arm_model(input_state_arm)
                action[0] = arm_action[0]
                action[1] = x
                action[3] = arm_action[1]
                action[5] = arm_action[2]
                action[7] = arm_action[3]
                action[9] = - 2.2 + (z * 1.3)

            """codice training da qui"""
            paddle_pos = np.array(state[11:14])
            ball_pos = np.array(state[17:20])

            distance = np.linalg.norm(paddle_pos - ball_pos)

            done = False

            if distance <= 0.3 and stance_chosen:

                for i in range(6):
                    input_state_paddle[i] = state[i + 17]

                input_state_paddle = torch.Tensor(input_state_paddle).to(device, dtype=torch.float32)

                if not wait_bounce_to_smash:

                    dont_wait_action = dont_wait_agent.calc_action(input_state_paddle)

                    action[9] = 1.5 - dont_wait_action[0]
                    action[10] = dont_wait_action[1]

                if wait_bounce_to_smash:

                    smash_action = smash_agent.calc_action(input_state_paddle)

                    action[9] = - 2.2 + (z * 1.3) + smash_action[0]
                    action[10] = smash_action[1]

        cli.send_joints(action)
        prev_state = state


def main():
    name = 'Example Client'
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
    python paddle_train.py name port host
    Default parameters:
     name: 'Example Client'
     port: client.DEFAULT_PORT
     host: 'localhost'

    To run the one simulation on the server, run this in 3 separate command shells:
    > python paddle_train.py player_A
    > python paddle_train.py player_B
    > python server.py

    To run a second simulation, select a different PORT on the server:
    > python paddle_train.py player_A 9544
    > python paddle_train.py player_B 9544
    > python server.py -port 9544    
    '''

    main()

