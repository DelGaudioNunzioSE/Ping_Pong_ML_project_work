# Machine Learning Project: Tennis Table
This project involved developing one or more neural networks to control a robotic arm in a virtual environment, with the goal of winning a ping pong competition against robotic arms operated by other university groups.

### Group 2 Members:
- Ciaravola Giosuè
- Conato Christian
- Del Gaudio Nunzio
- Garofalo Mariachiara

## How to Run:
First of all, to run the code, it is necessary to resolve some dependencies on files provided by the commission for the Project Work. Therefore, place the following files inside the project directory (at the same level as the file _insert_here.txt_): _channel.py_, _client.py_, _server.py_, _interface.txt_, _ball.urdf_, _robot.rdf_, _robot2.urdf_, _table.urdf_.\
\
To run the server:
`python .\server.py`\
\
If you want to check all the possible options for the server check _interface.txt_.


#### Use/test of the arm:
In order to **use the arm** in the environment, run the _arm_paddle_test.pyy_ with this command:\
`python .\train_test\arm_paddle_test.py`

#### Reinforcement Learning:
If you want to start a **reinforcement learning** session with the disengaged opponent, you need to start the server and connect two players:
- The opponent: `python .\train_test\auto_example.py`;
- The arm with paddles to train: `python .\train_test\paddle_train.py`;

### Supervised Learning:
If you want to start a **supervised learning** session for the arm model, firstly you need to build the dataset with:\
`python .\train_test\dataset_builder.py`\
\
and finally you can start:\
`python .\train_test\arm_supervised_train.py`

## Directory Contents

### nets
- **saved_models_arm**: Directory containing the arm model.
- **saved_models_dont_wait**: Directory containing the paddle model for lower balls.
- **saved_models_smash**: Directory containing the paddle model for higher balls.
- **arm_net.py**: File containing the architecture of the model that moves the arm, trained with supervised learning.
- **ddpg.py**: File containing the structure for learning using the DDPG algorithm, including declarations for the Actor and Critic and their respective weight update phases.
- **nets.py**: File containing the model structures used as Actor and Critic in the DDPG, trained via reinforcement learning.

### train_test
- **arm_learning_report.csv**: Report of the supervised learning with loss trend.
- **arm_paddle_test.py**: File containing the logic for using the networks to control the arm and paddle in their various types of movements.
- **arm_supervised_learning.py**: File containing the supervised learning for controlling the arm movements.
- **auto_example.py**: File containing the control logic for the opponent used in reinforcement training, which only plays to return the serve when it is directed towards them, and then disengages from the action.
- **dataset_builder.py**: File containing the randomization of positions for building the dataset needed for supervised learning.
- **dataset_file.csv**: Dataset obtained with a night of builder execution.
- **paddle_train.py**: File containing reinforcement learning for the two types of paddles, with alternating phases of episode recording and training.
- **plot.png**: Plot of the loss trend in the supervised learning.
- **plot_val.py**: A file useful for creating a plot to show the loss trend based on the report (.csv) that records the loss values during the epochs of supervised training.

### utilities
- **action_space.py**: File containing the action space (output) of all 3 networks, with declaration of upper and lower limits for each action, and denormalization for models using hyperbolic tangent.
- **dataset_loader.py**: File containing the class responsible for loading the dataset and splitting it into training, testing, and validation sets, providing their respective loaders.
- **early_stopping.py**: File containing the class responsible for monitoring the improvement of validation loss to apply early stopping in supervised learning.
- **noise.py**: File containing Ornstein-Uhlenbeck Action Noise used as noise during reinforcement learning, which follows a stochastic process that makes movements similar for a certain period of time, adding variability to the model's actions.
- **replay_memory.py**: File containing the class representing the replay buffer where to record transitions during reinforcement learning.
- **reward_calculator**: File containing the reward calculation to use during reinforcement training.
- **trajectory.py**: File containing the functions for trajectory calculation and finding the maximum point.
