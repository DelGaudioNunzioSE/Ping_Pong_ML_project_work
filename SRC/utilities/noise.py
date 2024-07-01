"""

    Machine Learning Project Work: Tennis Table Tournament
    Group 2:
        Ciaravola Giosuè - g.ciaravola3@studenti.unisa.it
        Conato Christian - c.conato@studenti.unisa.it
        Del Gaudio Nunzio - n.delgaudio5@studenti.unisa.it
        Garofalo Mariachiara - m.garofalo38@studenti.unisa.it

    ---------------------------------------------------------------

    noise.py

    File containing Ornstein-Uhlenbeck Action Noise used as noise
    during reinforcement learning, which follows a stochastic process
    that makes movements similar for a certain period of time,
    adding variability to the model's actions.

"""

import numpy as np


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated noise.

    Attributes:
        mu (np.array): Mean of the noise.
        sigma (np.array): Standard deviation of the noise.
        theta (float): Coefficient determining the speed of mean reversion.
        dt (float): Time step for discretization.
        x0 (np.array or None): Initial value of the noise process. Defaults to zero array.
    """
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        """
        Initialize the Ornstein-Uhlenbeck Action Noise.

        Args:
            mu (np.array): Mean of the noise.
            sigma (np.array): Standard deviation of the noise.
            theta (float, optional): Coefficient determining the speed of mean reversion. Default is 0.15.
            dt (float, optional): Time step for discretization. Default is 1e-2.
            x0 (np.array or None, optional): Initial value of the noise process. Default is None.
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        """
        Generate a sample of noise following the Ornstein-Uhlenbeck process.

        Returns:
            np.array: Generated noise sample.
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        self.x_prev = x

        return x

    def reset(self):
        """
        Reset the noise process to its initial state.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        """
        String representation of the OrnsteinUhlenbeckActionNoise object.
        """
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
