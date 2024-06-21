from client import Client, DEFAULT_PORT
import sys
from utilities.action_space import ActionSpace
import torch
import logging

from utilities.ddpg import DDPG

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

    agent = DDPG(gamma,
                 tau,
                 hidden_size,
                 cli.get_state().shape[0],
                 action_space,
                 checkpoint_dir="saved_models"
                 )

    agent.load_checkpoint()

    # Load the agents parameters
    agent.set_eval()

    while True:
        state = torch.Tensor(cli.get_state())

        action = agent.calc_action(state)

        cli.send_joints(action)


def main():
    name = 'DDPG test'
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
