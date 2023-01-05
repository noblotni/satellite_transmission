"""Satellite environment."""
from typing import List
from gym import Env
from gym import spaces
import numpy as np

# Define constants
# Modems constants
MOD_NB_LINKS = 3
MOD_BIN_FLOW = 3
MOD_SYMB_FLOW = 3
# Groups constants
GRP_NB_LINKS = 15
GRP_BANDWIDTH = 30


class SatelliteEnv(Env):
    """Custom environment to model the optimization problem."""

    def __init__(self, data: List):
        """Create an environment where to put satellite links."""
        super().__init__
        self.links = data
        self.nb_links = len(self.links)
        self.grp_mod_array = np.zeros((self.nb_links, self.nb_links, self.nb_links))
        self.state = np.array([(i, i) for i in range(self.nb_links)])
        self.set_group_modem_array()
        # Action space
        self.action_space = spaces.Box(low=0, high=self.nb_links, shape=(3), dtype=int)
        # Observation space
        self.observation_space = spaces.Box(
            low=0, high=self.nb_links, shape=(self.nb_links, 2), dtype=int
        )

    def set_group_modem_array(self):
        """Update the group modem array.

        It is an array with a 1 at the indice (i,j,k) if the
        the link k is affected to the modem j of the group i and
        0 otherwise.
        """
        self.grp_mod_array = np.zeros((self.nb_links, self.nb_links, self.nb_links))
        for i in range(self.nb_links):
            self.grp_mod_array[self.state[i, 0]][self.state[i, 1]][i] = 1

    def reward_function(self):
        # Number of modems
        nb_modems = np.sum(self.grp_mod_array)
        # Number of groups
        nb_grps = np.sum(np.sum(np.sum(self.grp_mod_array, axis=-1), axis=1) > 0)
        return -(nb_modems + nb_grps)

    def step(self, action):
        """Execute one time step within the environment."""
        state_before_action = np.copy(self.state)
        self.take_action(action)
        legal_move = self.is_legal_move()
        if not legal_move:
            self.state = state_before_action
            self.set_group_modem_array()
        reward = self.reward_function()
        return self.state, reward, False, {}

    def take_action(self, action):
        """Move one link from a case to another one."""
        self.state[action[0], :] = [action[1], action[2]]
        self.set_group_modem_array()

    def is_legal_move(self):
        if np.sum(self.grp_mod_array) != self.nb_links:
            return False
        indices_ones = np.where(self.grp_mod_array > 0)[0]
        modems_ok = self.check_modems(indices_ones)
        groups_ok = self.check_groups(indices_ones)
        return modems_ok and groups_ok

    def check_modems(self, used_modems):
        for (i, j, _) in used_modems:
            pass

    def check_groups(self, used_groups):
        pass

    def reset(self):
        """Reset the environment to an initial state."""
        self.state = np.array([(i, i) for i in range(self.nb_links)])
        self.set_group_modem_array()
        return self.state

    def render(self):
        """Render the environment to the screen."""
        pass
