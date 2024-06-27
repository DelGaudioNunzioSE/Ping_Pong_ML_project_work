import torch
from client import Client, DEFAULT_PORT
import sys
from server import AutoPlayerInterface, get_neutral_joint_position
import math

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

auto = AutoPlayerInterface()

def run(cli):

    action = get_neutral_joint_position()

    prev_state = cli.get_state()
    play = False

    while True:

        state = cli.get_state()

        if not prev_state[28] and state[28]:
            play = not play

        if state[21] > 0:
            play = False

        if not play:
            action = get_neutral_joint_position()
            action[2] += math.pi

        if play:
            action = auto.update(state)

        if not state[28]:
            action[2] = math.pi

        cli.send_joints(action)

        prev_state = state


def main():
    name = 'Auto'
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
