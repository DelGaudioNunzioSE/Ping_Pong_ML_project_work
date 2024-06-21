import torch
import numpy as np


class BaseActionSpace:
    def __init__(self, joint_ranges):
        self.JOINT_RANGES = joint_ranges
        self.low = np.array([low for low, high in self.JOINT_RANGES])
        self.high = np.array([high for low, high in self.JOINT_RANGES])
        self.shape = (len(self.JOINT_RANGES),)

    def rescale_action(self, action):
        """
        Rescales action values from the range [-1, 1] to the actual joint ranges.

        Arguments:
            action: A tensor of actions in the range [-1, 1].

        Returns:
            A tensor of actions scaled to the joint ranges.
        """
        low = torch.tensor(self.low, device=action.device)
        high = torch.tensor(self.high, device=action.device)
        return low + (high - low) * (action + 1) / 2


class ActionSpaceCart(BaseActionSpace):
    JOINT_RANGES = [
        (-0.3, 0.3),  # Joint 0: Translation y
        (-0.8, 0.8),  # Joint 1: Translation x
    ]

    def __init__(self):
        super().__init__(self.JOINT_RANGES)


class ActionSpaceArm(BaseActionSpace):
    JOINT_RANGES = [
        (0, np.pi * 2/5),  # Joint 3: Pitch primo
        (-3 * np.pi / 4, 3 * np.pi / 4),    # Joint 5: Pitch secondo
    ]

    def __init__(self):
        super().__init__(self.JOINT_RANGES)


class ActionSpacePaddle(BaseActionSpace):
    JOINT_RANGES = [
        (-3 * np.pi / 4, 3 * np.pi / 4),  # Joint 7: Pitch terzo
        (-3 * np.pi / 4, 3 * np.pi / 4),    # Joint 9: Pitch
        (-np.pi, np.pi)                     # Joint 10: Roll
    ]

    def __init__(self):
        super().__init__(self.JOINT_RANGES)
