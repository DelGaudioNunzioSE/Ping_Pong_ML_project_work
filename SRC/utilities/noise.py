import numpy as np
from math import sqrt


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
# and adapted to be synchronous with https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OUNoise:
    def __init__(self, action_dimension, dt=0.01, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.dt = dt # Time interval (delta t)
        self.mu = mu # average value around which the noise oscillates
        self.theta = theta # Return speed towards mu
        self.sigma = sigma # Standard deviation of noise
        self.reset()

    # Method to reset the initial state
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu #creates an array of size self.action dimension filled with 1 multiplied by self.mu

    # Method for generating noise
    def noise(self):
        x = self.state
        # Calculation of state increment (dx) using the Ornstein-Uhlenbeck equation
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    #allows you to dynamically adjust the noise level in the parameters to maintain a desired standard deviation in the action space.
    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    #Returns a current state of the standard deviation of the noise in the parameters.
    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    #Provides a string representation of the class, useful for debugging and printing information.
    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)


def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist
