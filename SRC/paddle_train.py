import torch
from client import Client, DEFAULT_PORT
import sys
from utilities.action_space import ActionSpaceArm, ActionSpacePaddleSmash, ActionSpacePaddleDontWait
from server import get_neutral_joint_position
import numpy as np
import math
from utilities.arm_net import ArmModel
from utilities.trajectory import trajectory, max_height_point, touch_the_net
from utilities.replay_memory import ReplayMemory, Transition
from utilities.noise import OrnsteinUhlenbeckActionNoise
from utilities.ddpg import DDPG, hard_update
from utilities.reward_calculator import calculate_paddle_reward

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: ", device)


GAMMA = 0.99
TAU = 0.01
HIDDEN_SIZE_PADDLE = (300, 200, 100)
ACTION_SPACE_SMASH = ActionSpacePaddleSmash()
ACTION_SPACE_DONT_WAIT = ActionSpacePaddleDontWait()
NUM_INPUTS_PADDLE = 8
TOTAL_EPOCH = 4
BATCH_S = 50
BATCH_DW = 50
REPLAY_SIZE_SMASH = 200
REPLAY_SIZE_DW = 200

smash_agent = DDPG(GAMMA,
                 TAU,
                 HIDDEN_SIZE_PADDLE,
                 NUM_INPUTS_PADDLE,
                 ACTION_SPACE_SMASH,
                 checkpoint_dir="saved_models_smash"
                 )

dont_wait_agent = DDPG(GAMMA,
                 TAU,
                 HIDDEN_SIZE_PADDLE,
                 NUM_INPUTS_PADDLE,
                 ACTION_SPACE_DONT_WAIT,
                 checkpoint_dir="saved_models_dont_wait"
                 )

noise_stddev = 1

# Initialize OU-Noise for the exploration
ou_noise_smash = OrnsteinUhlenbeckActionNoise(mu=np.zeros(ACTION_SPACE_SMASH.shape),
                                            sigma=float(noise_stddev) * np.ones(ACTION_SPACE_SMASH.shape))

# Initialize OU-Noise for the exploration
ou_noise_dont_wait = OrnsteinUhlenbeckActionNoise(mu=np.zeros(ACTION_SPACE_DONT_WAIT.shape),
                                            sigma=float(noise_stddev) * np.ones(ACTION_SPACE_DONT_WAIT.shape))

smash_memory = ReplayMemory(REPLAY_SIZE_SMASH)
dont_wait_memory = ReplayMemory(REPLAY_SIZE_DW)

HIDDEN_SIZE_ARM = (100, 50)
ACTION_SPACE_ARM = ActionSpaceArm()
NUM_INPUTS_ARM = 2

arm_model = ArmModel(HIDDEN_SIZE_ARM, NUM_INPUTS_ARM, ACTION_SPACE_ARM).to(device)

arm_model.load_checkpoint()

arm_model.eval()


def run(cli):

    tr_dw = 0
    tr_s = 0

    model_number_s = 0
    model_number_dw = 0

    hard_s = 0
    hard_dw = 0

    out = False
    wait_bounce_to_smash = False
    stance_chosen = False

    input_state_arm = np.zeros(NUM_INPUTS_ARM)
    input_state_paddle = np.zeros(NUM_INPUTS_PADDLE)

    action = get_neutral_joint_position()
    prev_state = cli.get_state()

    while True:
        if tr_s >= REPLAY_SIZE_SMASH:
            tr_s = 0

        if tr_dw >= REPLAY_SIZE_DW:
            tr_dw = 0

        # Each state:
        # - Read the state;
        # - Set neutral (changed if the model is activated);
        # - Set cart as far ahead as possible (changed if the model is activated);
        prev_state = cli.get_state()

        while tr_s < REPLAY_SIZE_SMASH and tr_dw < REPLAY_SIZE_DW:

            state = cli.get_state()

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
                if (prev_state[21] * state[21]) <= 0 and not touch_the_net(state):
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
                    action[9] = - 2.1 + (z * 1.3)

                """codice training da qui"""
                paddle_pos = np.array(state[11:14])
                ball_pos = np.array(state[17:20])

                distance = np.linalg.norm(paddle_pos - ball_pos)

                done = False

                if distance <= 0.3 and stance_chosen:

                    if not wait_bounce_to_smash:

                        while not done:

                            input_state_paddle[0] = state[9]
                            input_state_paddle[1] = state[10]

                            for i in range(6):
                                input_state_paddle[i] = state[i + 11]

                            input_state_paddle = torch.Tensor(input_state_paddle).to(device, dtype=torch.float32)

                            dont_wait_action = dont_wait_agent.calc_action(input_state_paddle, ou_noise_dont_wait)

                            action[9] = dont_wait_action[0]
                            action[10] = dont_wait_action[1]

                            cli.send_joints(action)

                            prev_state = state

                            state = cli.get_state()

                            paddle_pos = np.array(state[11:14])
                            ball_pos = np.array(state[17:20])

                            distance = np.linalg.norm(paddle_pos - ball_pos)

                            if (prev_state[21] * state[21] < 0) or distance > 0.3 or (prev_state[28] == 1 and state[28] == 0):
                                done = True

                            reward = calculate_paddle_reward(prev_state, state, done)

                            print("Reward dont_wait: ", reward)

                            next_input_state_paddle = np.zeros(NUM_INPUTS_PADDLE, dtype=np.float32)

                            next_input_state_paddle[0] = state[9]
                            next_input_state_paddle[1] = state[10]

                            for i in range(6):
                                next_input_state_paddle[i] = state[i + 11]

                            tr_dw += 1
                            print("DW Transition registered: ", tr_dw)
                            next_input_state_paddle = torch.Tensor(next_input_state_paddle).to(device, dtype=torch.float32)
                            dont_wait_action = torch.Tensor(dont_wait_action).to(device, dtype=torch.float32)
                            mask = torch.Tensor([done]).to(device, dtype=torch.float32)
                            reward = torch.Tensor([reward]).to(device, dtype=torch.float32)

                            dont_wait_memory.push(input_state_paddle, dont_wait_action, mask, next_input_state_paddle, reward)

                    if wait_bounce_to_smash:

                        while not done:

                            input_state_paddle[0] = state[9]
                            input_state_paddle[1] = state[10]

                            for i in range(6):
                                input_state_paddle[i] = state[i + 11]

                            input_state_paddle = torch.Tensor(input_state_paddle).to(device, dtype=torch.float32)

                            smash_action = smash_agent.calc_action(input_state_paddle, ou_noise_smash)

                            action[9] = smash_action[0]
                            action[10] = smash_action[1]

                            cli.send_joints(action)

                            prev_state = state

                            state = cli.get_state()

                            paddle_pos = np.array(state[11:14])
                            ball_pos = np.array(state[17:20])

                            distance = np.linalg.norm(paddle_pos - ball_pos)

                            if (prev_state[21] * state[21] < 0) or distance > 0.3 or (prev_state[28] == 1 and state[28] == 0):
                                done = True

                            reward = calculate_paddle_reward(prev_state, state, done)

                            print("Reward smash: ", reward)

                            next_input_state_paddle = np.zeros(NUM_INPUTS_PADDLE)

                            next_input_state_paddle[0] = state[9]
                            next_input_state_paddle[1] = state[10]

                            for i in range(6):
                                next_input_state_paddle[i] = state[i + 11]

                            tr_s += 1
                            print("S Transition registered: ", tr_s)
                            next_input_state_paddle = torch.Tensor(next_input_state_paddle).to(device, dtype=torch.float32)
                            smash_action = torch.Tensor(smash_action).to(device, dtype=torch.float32)
                            mask = torch.Tensor([done]).to(device, dtype=torch.float32)
                            reward = torch.Tensor([reward]).to(device, dtype=torch.float32)

                            smash_memory.push(input_state_paddle, smash_action, mask, next_input_state_paddle, reward)

            cli.send_joints(action)
            prev_state = state
            #print("ts: ", tr_s, "tdw: ", tr_dw, "tot: ", tr_ot,)

        if tr_s >= REPLAY_SIZE_SMASH:
            epoch = 0
            while epoch < TOTAL_EPOCH:
                count_batch = 0
                print("Epoch Smash: ", epoch)
                while count_batch < BATCH_S:
                    transitions = smash_memory.sample(1)
                    batch = Transition(*zip(*transitions))

                    # Update actor and critic according to the batch
                    smash_agent.update_params(batch)
                    count_batch += 1
                    # print("Batch: ", count_batch)
                epoch += 1
                hard_s += 1
            model_number_s += epoch

        if tr_dw >= REPLAY_SIZE_DW:
            epoch = 0
            while epoch < TOTAL_EPOCH:
                count_batch = 0
                print("Epoch Smash: ", epoch)
                while count_batch < BATCH_DW:
                    transitions = dont_wait_memory.sample(1)
                    batch = Transition(*zip(*transitions))

                    # Update actor and critic according to the batch
                    dont_wait_agent.update_params(batch)
                    count_batch += 1
                    # print("Batch: ", count_batch)
                epoch += 1
                hard_dw += 1
            model_number_dw += epoch

        if hard_s == 52:
            hard_s = 0
            hard_update(smash_agent.actor_target, smash_agent.actor)
            hard_update(smash_agent.critic_target, smash_agent.critic)

        if hard_dw == 52:
            hard_dw = 0
            hard_update(dont_wait_agent.actor_target, dont_wait_agent.actor)
            hard_update(dont_wait_agent.critic_target, dont_wait_agent.critic)

        if (model_number_s != 0) and (model_number_s % 16) == 0:
            smash_agent.save_checkpoint(model_number_s, "smash_agent")
            print("Saved smash_agent at epoch: ", model_number_s)

        if (model_number_dw != 0) and (model_number_dw % 16) == 0:
            dont_wait_agent.save_checkpoint(model_number_dw, "dont_wait_agent")
            print("Saved dont_wait_agent at epoch: ", model_number_dw)


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

