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


gamma = 0.99
tau = 0.01
hidden_size_cart = (76, 38)
hidden_size_arm = (152, 76)
hidden_size_paddle = (152, 76)
# action_space_cart = ActionSpaceCart()
action_space_paddle = ActionSpacePaddle()
action_space_arm = ActionSpaceArm()
auto = AutoPlayerInterface()
noise_stddev = 0.2
input_space_arm = 8

"""
cart_agent = DDPG(gamma,
             tau,
             hidden_size_cart,
             cli.get_state().shape[0],
             action_space_cart
             )
             """


# Initialize OU-Noise
#nb_actions_cart = action_space_cart.shape
#ou_noise_cart = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions_cart),
#                                        sigma=float(noise_stddev) * np.ones(nb_actions_cart))

nb_actions_arm = action_space_arm.shape
ou_noise_arm = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions_arm),
                                        sigma=float(noise_stddev) * np.ones(nb_actions_arm))

nb_actions_paddle = action_space_paddle.shape
ou_noise_paddle = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions_paddle),
                                               sigma=float(noise_stddev) * np.ones(nb_actions_paddle))


def run(cli):

    jp = get_neutral_joint_position()

    """
    arm_agent = DDPG(gamma,
                      tau,
                      hidden_size_arm,
                      input_space_arm,
                      action_space_arm
                      )
                      """

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
    out = False
    action = get_neutral_joint_position()


    while True:

        state = cli.get_state()

        # Game finished
        if not prev_state[28] and state[28]:
            action = get_neutral_joint_position()
            action[7] = math.pi*1/2
            action[9] = -math.pi*3/4
            out = False

        if state[21] < 0 and state[28] and not out and state[19] > 0:
            x, y = traiettoria(state)
            if x is not None and y is not None:
                if (prev_state[21] * state[21]) < 0 and ((x < -0.8 or x > 0.8) or (y < -0.1 or y > 1.3)):
                    print("va fuori")
                    action = get_neutral_joint_position()
                    out = True
                    if x <= 0:
                        x = 0.8
                    else:
                        x = -0.8
                else:
                    y += 0.35
                    y = max(-0.3, min(y, 0.3))
                    x = max(-0.8, min(x, 0.8))

                action[0] = y
                action[1] = x

            prev_state = state

            """
            input_state = np.zeros(input_space_arm)
            input_state[0] = state[3]
            input_state[1] = state[5]
            input_state[2] = state[11]
            input_state[3] = state[12]
            input_state[4] = state[13]
            input_state[5] = state[17]
            input_state[6] = state[18]
            input_state[7] = state[19]

            input_state = torch.Tensor(input_state)

            arm_action = arm_agent.calc_action(input_state, ou_noise_arm)
            action[3] = arm_action[0]
            action[5] = arm_action[1]
            """

        cli.send_joints(action)

        prev_state = state


        """
        action = jp

        # action = auto.update(state)

        choose_position(state, action)
        bx, by = traiettoria(state)

        count += 1

        if (prev_state[21] * state[21]) < 0 and state[28]:
            print(count, ": Predetta: x: ", bx, "  y: ", by)

        if state[19] < 0.1 and state[22] < 0 and state[28] and state[19] > 0:
            print(count,": Vera: x: ", state[17], "y: ", state[18], "z: ", state[19])
        """

        """
        state = torch.Tensor(state)

        # cart_action = cart_agent.calc_action(state, ou_noise_cart)
        # paddle_action = paddle_agent.calc_action(state, ou_noise_paddle)
        arm_action = arm_agent.calc_action(state, ou_noise_arm)

        # action[:2] = cart_action[:2]
        # action[-2:] = paddle_action[-2:]
        action[3] = arm_action[0]
        action[5] = arm_action[1]
        action[7] = arm_action[2]
        action[9] = 0
        action[10] = 0
        """

        # action = get_neutral_joint_position()




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
