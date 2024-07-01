"""

    Machine Learning Project Work: Tennis Table Tournament
    Group 2:
        Ciaravola Giosuè - g.ciaravola3#studenti.unisa.it
        Conato Christian - c.conato@studenti.unisa.it
        Del Gaudio Nunzio - n.delgaudio5@studenti.unisa.it
        Garofalo Mariachiara - m.garofalo38@studenti.unisa.it

    ---------------------------------------------------------------

    auto_example.py

    File containing the control logic for the opponent used in
    reinforcement training, which only plays to return the serve when
    it is directed towards them, and then disengages from the action.

"""

import sys
import os

# Get the current and parent directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)  # Add the parent directory to the system path

import torch
from client import Client, DEFAULT_PORT
import sys
from server import AutoPlayerInterface, get_neutral_joint_position
import math

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the AutoPlayerInterface
auto = AutoPlayerInterface()


def run(cli):
    """
    Main loop to control the robotic arm.

    Args:
        cli (Client): Client object to interact with the server.
    """

    action = get_neutral_joint_position()  # Get the neutral joint position

    prev_state = cli.get_state()  # Get the initial state
    play = False  # Initialize the play flag

    while True:

        state = cli.get_state()

        # Toggle play when the game start (according the serve)
        if not prev_state[28] and state[28]:
            play = not play

        # Stop playing if the ball is going away (serve done)
        if state[21] > 0:
            play = False

        if not play:
            action = get_neutral_joint_position()
            action[2] += math.pi  # Turn around if not playing
            action[1] = 0.8  # Go to the corner

        if play:
            action = auto.update(state)  # Turn again if the game is not playing

        if not state[28]:
            action[2] = math.pi
            action[1] = 0

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
