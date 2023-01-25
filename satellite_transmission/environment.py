"""Satellite environment."""
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
    def __init__(self, data: list):
        """Create an environment where to put satellite links."""
        super().__init__
        self.links = data
        self.nb_links = len(self.links)
        self.grp_mod_array = np.zeros((self.nb_links, self.nb_links, self.nb_links))
        self.state = np.array([(i, i) for i in range(self.nb_links)])
        self.action_shape = (3,)
        self.action_space = spaces.Box(low=np.zeros(self.action_shape), high=np.full(self.action_shape,self.nb_links-1), shape=self.action_shape, dtype=int)
        self.observation_shape = (self.nb_links, 2)
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape), high=np.full(self.observation_shape, self.nb_links), shape=self.observation_shape, dtype=int)

    def reward_function(self) -> float:
        """Compute the reward."""
        nb_modems = np.sum(self.grp_mod_array)
        nb_grps = np.sum(np.sum(np.sum(self.grp_mod_array, axis=-1), axis=1) > 0)
        return -(nb_modems + nb_grps)

    def step(self, action: np.ndarray) -> tuple:
        """Execute one time step within the environment."""
        state_before_action = np.copy(self.state)
        self.take_action(action)
        legal_move = self.is_legal_move()
        if not legal_move:
            self.state = state_before_action
        reward = self.reward_function()
        return self.state, reward, False, {}

    def take_action(self, action: np.ndarray):
        """Move one link from a case to another one."""
        print(action)
        self.state[action[0], :] = [action[1], action[2]]

    def is_legal_move(self) -> bool:
        """Check if the move is legal."""
        """if np.sum(self.grp_mod_array) != self.nb_links:
            return False"""
        modems_ok = self.check_modems()
        groups_ok = self.check_groups()
        return modems_ok and groups_ok

    def check_modems(self) -> bool:
        """Check if the modems are legal."""
        duplicate = []
        for s in self.state:
            if not np.any(s == duplicate):#if s not in np.array(duplicate):
                indices = set(np.where(self.state == s)[0])
                links_indices = [self.links[indice] for indice in indices]
                binary_flow = np.sum([link["binary_flow"] for link in links_indices])
                symbole_flow = np.sum([link["symbol_flow"] for link in links_indices])
            if binary_flow > MOD_BIN_FLOW or symbole_flow > MOD_SYMB_FLOW or len(indices) > MOD_NB_LINKS:
                return False
            duplicate.append(s)
        return True

    def check_groups(self) -> bool:
        """ Check if the groups are legal"""
        duplicate = []
        for s in self.state:
            if not np.any(s == duplicate):
                indices = set(np.where(self.state[:,0] == s[0])[0])
                links_indices = [self.links[indice] for indice in indices]
                bandwidth = np.sum([link["bandwidth"] for link in links_indices])
                inverse_binary_flow = np.sum([link["inverse_binary_flow"] for link in links_indices])
                min_group_inverse_binary_flow = np.min([link["group_inverse_binary_flow"] for link in links_indices])
            if bandwidth > GRP_BANDWIDTH or len(indices) > GRP_NB_LINKS or inverse_binary_flow < min_group_inverse_binary_flow:
                return False
            duplicate.append(s)
        return True

    def reset(self) -> np.ndarray:
        """Reset the environment to an initial state."""
        self.state = np.array([(i, i) for i in range(self.nb_links)])
        return self.state