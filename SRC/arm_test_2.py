import torch
from client import Client, DEFAULT_PORT
import sys
from utilities.action_space import ActionSpaceArm
from server import get_neutral_joint_position
import numpy as np
import math
from utilities.arm_net import ArmModel
from utilities.trajectory import trajectory

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: ", device)

HIDDEN_SIZE_ARM = (100, 50)
ACTION_SPACE_ARM = ActionSpaceArm()
NUM_INPUTS = 2

arm_model = ArmModel(HIDDEN_SIZE_ARM, NUM_INPUTS, ACTION_SPACE_ARM).to(device)

arm_model.load_checkpoint()

arm_model.eval()

def run(cli):

    out = False
    wait_bounce = False
    smash = False
    dont_smash = False
    stance_chosen = False
    input_state = np.zeros(NUM_INPUTS)
    action = get_neutral_joint_position()
    prev_state = cli.get_state()

    while True:

        # Each state:
        # - Read the state;
        # - Set neutral (changed if the model is activated);
        # - Set cart as far ahead as possible (changed if the model is activated);

        state = cli.get_state()

        # Game finished:
        # - Reset 'out';
        if not prev_state[28] and state[28]:

            print("FINISH!")
            x = 0
            y = 0.3
            z = 0.2

            input_state[0] = y + 0.4
            input_state[1] = z + 0.4

            input_state = torch.Tensor(input_state)

            arm_action = arm_model(input_state)
            action[0] = arm_action[0]
            action[1] = x
            action[3] = arm_action[1]
            action[5] = arm_action[2]
            action[7] = arm_action[3] - math.pi / 6
            action[9] = math.pi * 3 / 4

            out = False
            wait_bounce = False
            smash = False
            dont_smash = False
            stance_chosen = False

        elif state[21] > 0 and prev_state[18] > 1.2:

            x = 0
            y = 0.3
            z = 0.2

            input_state[0] = y + 0.4
            input_state[1] = z + 0.4

            input_state = torch.Tensor(input_state)

            arm_action = arm_model(input_state)
            action[0] = arm_action[0]
            action[1] = x
            action[3] = arm_action[1]
            action[5] = arm_action[2]
            action[7] = arm_action[3] - math.pi / 6
            action[9] = math.pi * 3 / 4

            wait_bounce = False
            smash = False
            dont_smash = False
            stance_chosen = False

        # Activation:
        # - If the ball is coming to us (negative ball-y-velocity);
        # - And the game is playing;
        # - And the ball (cat) is on the table (negative ball-z-position)
        if state[21] < 0 and state[28] and not out and state[19] > 0:
            # Calculate the trajectory to check if the ball go on our side of the table
            z = 0.1
            x, y = trajectory(state, z)
            if x is not None and y is not None:
                if (prev_state[21] * state[21]) < 0 and ((x < -0.7 or x > 0.7) or (y < -0.2 or y > 1.3)):
                    print("va fuori")
                    action = get_neutral_joint_position()
                    out = True
                    if x <= 0:
                        action[1] = 0.8
                    else:
                        action[1] = -0.8
                else:
                    # print("Sono arrivato qui con y = ", y)
                    if y >= 0.3 and not wait_bounce and not stance_chosen:

                        print("WAIT!")
                        wait_bounce = True
                        action[0] = -0.3
                        action[9] = 0

                    elif not wait_bounce and not stance_chosen:

                        print("DON'T WAIT!")
                        stance_chosen = True
                        input_state[0] = y + 0.4
                        input_state[1] = z + 0.4

                        input_state = torch.Tensor(input_state)

                        arm_action = arm_model(input_state)
                        action[0] = arm_action[0]
                        action[3] = arm_action[1]
                        action[5] = arm_action[2]
                        action[7] = arm_action[3] - math.pi / 6
                        action[9] = math.pi * 3 / 4

                    if wait_bounce and prev_state[22] < 0 and state[22] > 0 and prev_state[18] < 1.2:

                        # dont_smash = True

                        z_smash = 1.4
                        while not smash and not dont_smash:
                            x_smash, y_smash = trajectory(state, z_smash)
                            if x_smash is not None and y_smash is not None:
                                if y_smash <= 0.1:
                                    smash = True
                                    y = y_smash
                                    x = x_smash
                                    z = z_smash
                            if z_smash <= 0.5:
                                dont_smash = True
                            else:
                                z_smash -= 0.1

                        if smash and not stance_chosen:

                            stance_chosen = True
                            print("SMASH!")
                            input_state[0] = y + 0.2
                            input_state[1] = z - 0.35

                            input_state = torch.Tensor(input_state)

                            arm_action = arm_model(input_state)
                            action[0] = arm_action[0]
                            action[3] = arm_action[1]
                            action[5] = arm_action[2]
                            action[7] = arm_action[3]
                            action[9] = - math.pi * 5 / 8

                        if dont_smash and not stance_chosen:

                            stance_chosen = True
                            print("NORMAL!")
                            z = 0.05
                            x, y = trajectory(state, z)

                            if y > -0.2:

                                print("TABLE")

                                input_state[0] = y + 0.1
                                input_state[1] = z + 0.15

                                input_state = torch.Tensor(input_state)

                                arm_action = arm_model(input_state)
                                action[0] = arm_action[0]
                                action[3] = arm_action[1]
                                action[5] = arm_action[2]
                                action[7] = arm_action[3]
                                action[9] = math.pi * 1/10

                            else:

                                print("OUT TABLE")

                                input_state[0] = y + 0.05
                                input_state[1] = z + 0.1

                                input_state = torch.Tensor(input_state)

                                arm_action = arm_model(input_state)
                                action[0] = arm_action[0]
                                action[3] = arm_action[1]
                                action[5] = arm_action[2]
                                action[7] = arm_action[3] - math.pi / 6
                                action[9] = math.pi * 2 / 4

                    action[1] = x

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
