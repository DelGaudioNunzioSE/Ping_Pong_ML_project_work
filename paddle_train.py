import torch
import logging
from client import Client, DEFAULT_PORT
import sys
from utilities.action_space import ActionSpacePaddle, ActionSpaceArm
from utilities.replay_memory import ReplayMemory, Transition
from utilities.ddpg import DDPG, hard_update
from server import AutoPlayerInterface, get_neutral_joint_position
from utilities.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
from utilities.trajectory import trajectory
from utilities.reward_calculator import calculate_arm_reward, calculate_paddle_reward
import math

GAMMA = 0.99
TAU = 0.01
HIDDEN_SIZE_ARM = (400, 300)
HIDDEN_SIZE_PADDLE = (400, 300)
ACTION_SPACE_PADDLE = ActionSpacePaddle()
ACTION_SPACE_ARM = ActionSpaceArm()
AUTO = AutoPlayerInterface()
INPUT_SPACE_PADDLE = 15
INPUT_SPACE_ARM = 8
REPLAY_SIZE = 2500  # Replay buffer dimension (5 minutes for registration section)
BATCH_SIZE = 124
TOTAL_EPOCH = 20


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
                 ACTION_SPACE_ARM,
                 checkpoint_dir="saved_models_arm"
                 )

arm_agent.load_checkpoint()

paddle_agent = DDPG(GAMMA,
                 TAU,
                 HIDDEN_SIZE_PADDLE,
                 INPUT_SPACE_PADDLE,
                 ACTION_SPACE_PADDLE,
                 checkpoint_dir="saved_models_paddle"
                 )

memory = ReplayMemory(REPLAY_SIZE)


def run(cli):
    model_number = 0
    hard = 0
    noise_stddev = 0.4

    # Initialize OU-Noise for the exploration
    ou_noise_arm = OrnsteinUhlenbeckActionNoise(mu=np.zeros(ACTION_SPACE_PADDLE.shape),
                                                sigma=float(noise_stddev) * np.ones(ACTION_SPACE_PADDLE.shape))

    transition_registered = 0
    while True:
        prev_state = cli.get_state()
        while transition_registered < REPLAY_SIZE:
            state = cli.get_state()

            # Game started
            if not prev_state[28] and state[28]:
                action = get_neutral_joint_position()
                action[7] = math.pi * 1 / 2
                action[9] = -math.pi * 3 / 4
                out = False

            # Activation:
            # - if the ball is coming to the robot
            # - and the game is playing
            # - and the ball in not going out
            # - and the ball is over the table
            if state[21] < 0 and state[28] and not out and state[19] > 0:
                x, y = trajectory(state)
                if x is not None and y is not None:
                    # Check if the ball is going out of our side of the table
                    if (prev_state[21] * state[21]) < 0 and ((x < -0.8 or x > 0.8) or (y < -0.1 or y > 1.3)):
                        action = get_neutral_joint_position()
                        out = True
                        if x <= 0:
                            x = 0.8
                        else:
                            x = -0.8
                        action[1] = x
                        cli.send_joints(action)
                    else:
                        # episode start
                        while True:
                            done = 0.0

                            input_state_arm = np.zeros(INPUT_SPACE_ARM, dtype=np.float32)
                            input_state_arm[0] = state[3]
                            input_state_arm[1] = state[5]
                            input_state_arm[2] = state[11]
                            input_state_arm[3] = state[12]
                            input_state_arm[4] = state[13]
                            input_state_arm[5] = state[17]
                            input_state_arm[6] = state[18]
                            input_state_arm[7] = state[19]

                            input_state_arm = torch.Tensor(input_state_arm).to(device)

                            arm_action = arm_agent.calc_action(input_state_arm).to(torch.float32)

                            action[3] = arm_action[0]
                            action[5] = arm_action[1]

                            y += 0.35
                            y = max(-0.3, min(y, 0.3))
                            x = max(-0.8, min(x, 0.8))
                            action[0] = y
                            action[1] = x

                            paddle_pos = np.array(state[11:14])
                            ball_pos = np.array(state[17:20])

                            # Calculate the Euclidean distance between the ball and the center of the paddle
                            distance_paddle_ball = np.linalg.norm(paddle_pos - ball_pos)

                            # activation paddle
                            if distance_paddle_ball <= 0.7:
                                # print("Transition_registered: ", transition_registered)
                                input_state_paddle = np.zeros(INPUT_SPACE_PADDLE, dtype=np.float32)
                                input_state_paddle[0] = state[7]
                                input_state_paddle[1] = state[9]
                                input_state_paddle[2] = state[10]
                                input_state_paddle[3] = state[11]
                                input_state_paddle[4] = state[12]
                                input_state_paddle[5] = state[13]
                                input_state_paddle[6] = state[14]
                                input_state_paddle[7] = state[15]
                                input_state_paddle[8] = state[16]
                                input_state_paddle[9] = state[17]
                                input_state_paddle[10] = state[18]
                                input_state_paddle[11] = state[19]
                                input_state_paddle[12] = state[20]
                                input_state_paddle[13] = state[21]
                                input_state_paddle[14] = state[22]

                                input_state_paddle = torch.Tensor(input_state_paddle).to(device)

                                paddle_action = paddle_agent.calc_action(input_state_paddle).to(torch.float32)

                                action[7] = paddle_action[0]
                                action[9] = paddle_action[1]
                                action[10] = paddle_action[2]

                                cli.send_joints(action)
                                next_state = cli.get_state()

                                # Check if state is terminal
                                # - if the ball return back
                                # - or we miss it so the distance is high
                                # - or the game end
                                if next_state[21] > 0 or distance_paddle_ball > 0.7 or (state[28] == 1 and next_state[28] == 0):
                                    done = 1.0

                                reward = calculate_paddle_reward(state, next_state)
                                transition_registered += 1
                                print("Reward: ", reward)

                                next_input_state_paddle = np.zeros(INPUT_SPACE_PADDLE, dtype=np.float32)

                                next_input_state_paddle[0] = next_state[7]
                                next_input_state_paddle[1] = next_state[9]
                                next_input_state_paddle[2] = next_state[10]
                                next_input_state_paddle[3] = next_state[11]
                                next_input_state_paddle[4] = next_state[12]
                                next_input_state_paddle[5] = next_state[13]
                                next_input_state_paddle[6] = next_state[14]
                                next_input_state_paddle[7] = next_state[15]
                                next_input_state_paddle[8] = next_state[16]
                                next_input_state_paddle[9] = next_state[17]
                                next_input_state_paddle[10] = next_state[18]
                                next_input_state_paddle[11] = next_state[19]
                                next_input_state_paddle[12] = next_state[20]
                                next_input_state_paddle[13] = next_state[21]
                                next_input_state_paddle[14] = next_state[22]

                                paddle_action = torch.Tensor(paddle_action).to(device)
                                mask = torch.Tensor([done]).to(device)
                                next_input_state_paddle = torch.Tensor(next_input_state_paddle).to(device)
                                reward = torch.Tensor([reward]).to(device)

                                memory.push(input_state_paddle, paddle_action, mask, next_input_state_paddle, reward)

                            else:
                                cli.send_joints(action)

                            if done:
                                print("Fine: ", transition_registered)
                                break

            prev_state = state

        epoch = 0
        transition_registered = 0
        while epoch < TOTAL_EPOCH:
            count_batch = 0
            print("Epoch: ", epoch)
            while count_batch < BATCH_SIZE:
                transitions = memory.sample(1)
                batch = Transition(*zip(*transitions))

                # Update actor and critic according to the batch
                arm_agent.update_params(batch)
                count_batch += 1
                #print("Batch: ", count_batch)
            epoch += 1

        if noise_stddev > 0.1:
            noise_stddev -= 0.01
            print("Std-dev: ", noise_stddev)
            # Initialize OU-Noise for the exploration
            ou_noise_arm = OrnsteinUhlenbeckActionNoise(mu=np.zeros(ACTION_SPACE_ARM.shape),
                                                        sigma=float(noise_stddev) * np.ones(ACTION_SPACE_ARM.shape))

        hard_update(arm_agent.actor_target, arm_agent.actor)
        hard_update(arm_agent.critic_target, arm_agent.critic)
        model_number += epoch
        arm_agent.save_checkpoint(model_number, "arm_model_2")
        print("Saved arm_model at epoch: ", model_number)


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
