import torch
import time
from client import Client, JOINTS, DEFAULT_PORT
import sys
from utilities.action_space import ActionSpaceCart, ActionSpaceArm, ActionSpacePaddle
from utilities.ddpg import DDPG
from server import AutoPlayerInterface, get_neutral_joint_position
from utilities.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
import math

DT = 1.0/50

def traiettoria(state, target_z=0.1):
    bx, by, bz = state[17:20]
    vx, vy, vz = state[20:23]
    g = 9.81  # Accelerazione gravitazionale

    # Risolviamo l'equazione quadratica per trovare il tempo t in cui z(t) = target_z
    a = -0.5 * g
    b = vz
    c = bz - target_z

    # Calcoliamo il discriminante
    discriminante = b**2 - 4 * a * c

    if discriminante < 0:
        # La pallina non raggiunge mai z = target_z
        return None, None

    # Calcoliamo i tempi possibili
    t1 = (-b + math.sqrt(discriminante)) / (2 * a)
    t2 = (-b - math.sqrt(discriminante)) / (2 * a)

    # Prendiamo il tempo positivo
    t = max(t1, t2)

    if t < 0:
        # Entrambi i tempi sono negativi, la pallina non raggiunge mai z = target_z
        return None, None

    # Calcoliamo le posizioni x e y in cui la pallina raggiunge z = target_z
    x = bx + vx * t
    y = by + vy * t

    return x, y



def choose_position(state, jp):
    px, py, pz = state[11:14]
    bx, by, bz = state[17:20]
    vx, vy, vz = state[20:23]
    dist = math.hypot(px - bx, py - by, pz - bz)
    vel = math.hypot(vx, vy, vz)
    if state[27] or vel < 0.05:
        jp[1] = bx
        return
    extra_y = 0.0
    if dist < vel * 1.5 * DT:
        extra_y = 0.3
    d = 0.05
    g = 9.81
    while vz > 0 or bz + d * vz >= pz:
        bx += d * vx
        by += d * vy
        bz += d * vz
        vz -= d * g
    jp[1] = bx
    dy = py - state[0]
    jp[0] = by - dy + extra_y
    jp[10] -= (bx * 0.3 + vx * 0.01)

gamma = 0.99
tau = 0.01
hidden_size_cart = (76, 38)
hidden_size_arm = (152, 76)
hidden_size_paddle = (152, 76)
# action_space_cart = ActionSpaceCart()
action_space_paddle = ActionSpacePaddle()
# action_space_arm = ActionSpaceArm()
auto = AutoPlayerInterface()
noise_stddev = 0.5

"""
cart_agent = DDPG(gamma,
             tau,
             hidden_size_cart,
             cli.get_state().shape[0],
             action_space_cart
             )
             """

"""
arm_agent = DDPG(gamma,
                  tau,
                  hidden_size_arm,
                  cli.get_state().shape[0],
                  action_space_arm
                  )
                  """



"""
# Initialize OU-Noise
nb_actions_cart = action_space_cart.shape
ou_noise_cart = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions_cart),
                                        sigma=float(noise_stddev) * np.ones(nb_actions_cart))

nb_actions_arm = action_space_arm.shape
ou_noise_arm = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions_arm),
                                        sigma=float(noise_stddev) * np.ones(nb_actions_arm))
                                        """

nb_actions_paddle = action_space_paddle.shape
ou_noise_paddle = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions_paddle),
                                               sigma=float(noise_stddev) * np.ones(nb_actions_paddle))


def run(cli):

    jp = get_neutral_joint_position()

    """
    paddle_agent = DDPG(gamma,
                        tau,
                        hidden_size_paddle,
                        cli.get_state().shape[0],
                        action_space_paddle
                        )
                        """

    count = 0
    prev_state = cli.get_state()
    while True:

        state = cli.get_state()

        action = jp

        # action = auto.update(state)

        choose_position(state, action)
        bx, by = traiettoria(state)

        count += 1

        if (prev_state[21] * state[21]) < 0 and state[28]:
            print(count, ": Predetta: x: ", bx, "  y: ", by)

        if state[19] < 0.1 and state[22] < 0 and state[28] and state[19] > 0:
            print(count,": Vera: x: ", state[17], "y: ", state[18], "z: ", state[19])

        # state = torch.Tensor(state)

        # cart_action = cart_agent.calc_action(state, ou_noise_cart)
        # paddle_action = paddle_agent.calc_action(state, ou_noise_paddle)
        # arm_action = arm_agent.calc_action(state, ou_noise_arm)

        # action[:2] = cart_action[:2]
        # action[-2:] = paddle_action[-2:]
        # action[3] = arm_action[0]
        # action[5] = arm_action[1]
        # action[7] = arm_action[2]

        # action = get_neutral_joint_position()

        cli.send_joints(action)

        prev_state = state

        """
        count += 1
        print("Count: ",count)
        if count > 300:
            time.sleep(10)
            count = 0
            """

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
