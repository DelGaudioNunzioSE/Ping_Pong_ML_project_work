from client import Client, DEFAULT_PORT
import sys
from utilities.action_space import ActionSpace
import torch
import logging
from utilities.replay_memory import ReplayMemory, Transition
from server import AutoPlayerInterface
from utilities.reward_calculator import calculate_reward
import time

from utilities.ddpg import DDPG, hard_update

# Create logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))


def run(cli):
    # Some parameters, which are not saved in the network files
    gamma = 0.99  # discount factor for reward (default: 0.99)
    tau = 0.001  # discount factor for model (default: 0.001)
    hidden_size = (400, 300)  # size of the hidden layers (Deepmind: 400 and 300; OpenAI: 64)
    action_space = ActionSpace()
    replay_size = 501000
    total_time_step = 500000
    auto = AutoPlayerInterface()
    total_epoch = 5000
    batch_size = 124

    agent = DDPG(gamma,
                 tau,
                 hidden_size,
                 cli.get_state().shape[0],
                 action_space,
                 checkpoint_dir="saved_models"
                 )

    # Initialize replay memory
    memory = ReplayMemory(replay_size)

    # TODO: initialize noise

    # Define counters and other variables
    time_step = 0
    epoch = 0

    while True:
        while time_step <= total_time_step:
            state = cli.get_state()

            while True:
                done = 0.0
                print("Time_step", time_step)
                action = auto.update(state)
                cli.send_joints(action)
                next_state = cli.get_state()

                # Check if state is terminal
                if state[28] == 1 and next_state[28] == 0:
                    if state[34] == next_state[34] and state[35] == next_state[35]:
                        # Wait for the game to finish and update the scores
                        current_score = next_state[34]
                        opponent_score = next_state[35]

                        while True:
                            next_state = cli.get_state()
                            if next_state[34] != current_score or next_state[35] != opponent_score:
                                break

                    done = 1.0

                # Calculate reward
                reward = calculate_reward(state, next_state)
                time_step += 1

                state = torch.Tensor(state).to(device)
                action = torch.Tensor(action).to(device)
                mask = torch.Tensor([done]).to(device)
                next_state = torch.Tensor(next_state).to(device)
                reward = torch.Tensor([reward]).to(device)

                memory.push(state, action, mask, next_state, reward)

                state = next_state.cpu().numpy()

                if done:
                    print("fine")
                    break

        while epoch < total_epoch:
            count_batch = 0
            print("Epoca", epoch)
            while count_batch < batch_size:
                transitions = memory.sample(1)
                batch = Transition(*zip(*transitions))

                # Update actor and critic according to the batch
                agent.update_params(batch)
                count_batch += 1
                print("Batch", count_batch)
                if (epoch+1) % 500 == 0:
                    hard_update(agent.actor_target, agent.actor)
                    hard_update(agent.critic_target, agent.critic)

            if (epoch+1) % 1000 == 0:
                agent.save_checkpoint(epoch)
                # agent.save_checkpoint(time_step, memory)
                logger.info('Saved model {} at end_time {}'.format(time_step, time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

            epoch += 1

        logger.info('Stopping training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))


def main():
    name = 'DDPG train'
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
